import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    # Initialize wandb (mirrors rl_reward.py so code runs log the same panels)
    wandb.init(
        project=config.wandb.get("project"),
        entity=config.wandb.get("entity"),
        id=config.wandb.get("run_id"),
        resume=config.wandb.get("resume"),
        name=config.wandb.get("run_name"),
        reinit=True,
    )

    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name


    if config.experiment.function == "train":
        shrink = config.training.shrink
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset

    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset


    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]

    def dr_grpo_normalize(lst):
        mean = sum(lst) / len(lst)
        return [x - mean for x in lst]




    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst


    # Compute decoding metrics before step_map/commit_counts are cleared.
    # Code rollouts carry step_map; commit_counts/expected_wrong_commits may be
    # absent -> .get(...,[]) defaults them, matching rl_reward.py (no crash).
    unique_steps_list = []
    tokens_per_step_list = []
    commit_count_list = []
    ewc_cumsum_list = []
    ewc_mean_list = []
    for rec in data:
        for sm in rec.get("step_map", []):
            if sm:
                n_unique = len(set(sm))
                unique_steps_list.append(n_unique)
                tokens_per_step_list.append(len(sm) / n_unique)
        for cc_list in rec.get("commit_counts", []):
            if cc_list:
                commit_count_list.extend(cc_list)
        for ewc_list in rec.get("expected_wrong_commits", []):
            if ewc_list:
                ewc_cumsum_list.append(sum(ewc_list))
                ewc_mean_list.extend(ewc_list)
    avg_unique_steps = sum(unique_steps_list) / len(unique_steps_list) if unique_steps_list else 0.0
    avg_tokens_per_step = sum(tokens_per_step_list) / len(tokens_per_step_list) if tokens_per_step_list else 0.0
    commit_count_mean = sum(commit_count_list) / len(commit_count_list) if commit_count_list else 0.0
    ewc_mean = sum(ewc_mean_list) / len(ewc_mean_list) if ewc_mean_list else 0.0
    ewc_cumsum_mean = sum(ewc_cumsum_list) / len(ewc_cumsum_list) if ewc_cumsum_list else 0.0

    # Advantage normalization method (z_score default, dr_grpo optional) — now
    # honored on the code path too (previously hard-wired to z_score).
    normalize_method = OmegaConf.select(config, "training.normalize_method", default="z_score")


    response_length_list = []
    num_task   = 0
    num_correct_task = 0
    final_data = []
    # Track how many groups are filtered out (no-variance groups -> all-zero rewards)
    dropped_groups = 0
    kept_groups = 0
    for i in range(len(data)):
        response_length_list = response_length_list + data[i]["response_length"]
        acc_list = []
        for x in data[i]["correctness"]:
            acc_list.append(sum(x))            # per-candidate passed-test count
            num_correct_task += all(x)         # task correct iff ALL tests pass
            num_task += 1
        lengths = data[i]["response_length"]

        # NOTE: length-truncation penalty is intentionally NOT applied for code
        # tasks (unlike rl_reward.py:143-147) — per request.

        if normalize_method == "z_score":
            rewards = z_score_normalize(acc_list)
        elif normalize_method == "dr_grpo":
            rewards = dr_grpo_normalize(acc_list)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
        data[i]["rewards"] = rewards

        if config.experiment.function == "train":

            if all(x == 0 for x in rewards):
                dropped_groups += 1
                continue
            kept_groups += 1

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                final_data.append(data_i)

        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []
            if "commit_counts" in data[i]:
                data[i]["commit_counts"] = []
            if "expected_wrong_commits" in data[i]:
                data[i]["expected_wrong_commits"] = []


    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)


    import os

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")

        acc = num_correct_task / num_task if num_task else 0   # all-tests-pass rate
        avg_len = sum(response_length_list)/len(response_length_list)
        total_groups = len(data)
        reward_mean = acc
        dropped_rate = dropped_groups / total_groups if total_groups else 0.0

        output_text = f"train step: {config.experiment.current_epoch}  "

        if config.experiment.function == "train":
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}"
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}"
        if avg_unique_steps > 0 or commit_count_mean > 0:
            output_text += f"  avg_unique_steps: {avg_unique_steps:.2f}  tokens_per_step: {avg_tokens_per_step:.2f}  commit_count_mean: {commit_count_mean:.2f}  ewc_mean: {ewc_mean:.4f}  ewc_cumsum: {ewc_cumsum_mean:.4f}"
        save_and_print(output_text)

        metrics_data = {
            "acc": acc,
            "avg_len": avg_len,
            "reward_mean": reward_mean,
            "epoch": config.experiment.current_epoch,
            "mode": config.experiment.function,
            "prompts_total": total_groups,
            "prompts_kept": kept_groups,
            "prompts_dropped": dropped_groups,
            "prompts_drop_rate": dropped_rate,
            "avg_unique_steps": avg_unique_steps,
            "avg_tokens_per_step": avg_tokens_per_step,
            "commit_count_mean": commit_count_mean,
            "ewc_mean": ewc_mean,
            "ewc_cumsum": ewc_cumsum_mean,
        }

        metrics_file_path = f"../{project_name}/temp_data/temp_metrics.json"

        with open(metrics_file_path, "w", encoding="utf-8") as fm:
            json.dump(metrics_data, fm, indent=2)

        # Log to wandb
        if config.experiment.function == "evaluation":
            prefix = "eval"
            remask = config.evaluation.remasking_strategy
            metric_prefix = f"{prefix}/{remask}"

            wandb_log_dict = {
                f"{metric_prefix}/acc": acc,
                f"{metric_prefix}/avg_length": avg_len,
                f"{metric_prefix}/avg_unique_steps": avg_unique_steps,
                f"{metric_prefix}/avg_tokens_per_step": avg_tokens_per_step,
                f"{metric_prefix}/commit_count_mean": commit_count_mean,
                f"{metric_prefix}/ewc_mean": ewc_mean,
                f"{metric_prefix}/ewc_cumsum": ewc_cumsum_mean,
            }

            wandb.log(wandb_log_dict, step=config.experiment.current_epoch)

    wandb.finish()
