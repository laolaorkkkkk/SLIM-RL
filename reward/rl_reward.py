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

    # Initialize wandb
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.get("entity"),
        id=config.wandb.run_id,
        resume=config.wandb.resume,
        name=config.wandb.run_name,
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


    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    async def get_correctness():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = []
        for i in range(len(index_list)):
            tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
        results = await asyncio.gather(*tasks)
        return results

    correctness_list = asyncio.run(get_correctness())
    for i in range(len(index_list)):
        index_i = index_list[i]
        data[index_i]["correctness"].append(correctness_list[i])



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



    # Compute decoding metrics before step_map/commit_counts are cleared
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

    final_data = []
    # Track how many groups are filtered out 
    dropped_groups = 0
    kept_groups = 0
    for i in range(len(data)):
        correctness = data[i]["correctness"]
        lengths = data[i]["response_length"]

        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness[j] = False
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness[j] = False
        
        
        normalize_method = OmegaConf.select(config, "training.normalize_method", default="z_score")
        if normalize_method == "z_score":
            rewards = z_score_normalize(correctness)
        elif normalize_method == "dr_grpo":
            rewards = dr_grpo_normalize(correctness)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
        #rewards = [int(b) for b in correctness]

        data[i]["rewards"] = rewards # (n,)
        
        if config.experiment.function == "train":

            proportion = sum(correctness) / len(correctness)
            if proportion > 0.8 or proportion < 0.2:
                dropped_groups += 1
                continue
            kept_groups += 1

            for j in range(len(rewards)):
                #if rewards[j] == 0:
                #    continue
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                data_i["group_id"] = i
                final_data.append(data_i)
        
        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []
            data[i]["commit_counts"] = []
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
        
        acc = sum(correctness_list)/len(correctness_list)
        avg_len = sum(response_length_list)/len(response_length_list)
        total_groups = len(data)
        dropped_rate = dropped_groups / total_groups 

        output_text = f"train step: {config.experiment.current_epoch}  "
        
        if config.experiment.function == "train":
            strategy = config.rollout.remasking_strategy
            reward_mean = acc

            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}"
        else:
            strategy = config.evaluation.remasking_strategy
            reward_mean = acc
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
