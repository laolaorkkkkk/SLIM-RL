# =====================================================================
# Qwen2.5-7B-Instruct (and any standard Qwen2 / autoregressive HF causal LM)
# eval sampler for the dllm eval pipeline.
#
# WHY THIS FILE EXISTS: dllm's sample/*_sample.py are all DIFFUSION samplers
# (block denoise + remasking). Qwen2.5 is autoregressive, so it needs its own
# generation kernel. Everything ELSE — dataset load, 2-node sharding, ChatML
# prompt template, boxed/code answer extraction, and the per-record output
# schema — is copied VERBATIM from sample/dream_sample.py so the downstream
# stages (reward/execute.py -> reward/aggregate_data.py -> reward/reward.py)
# work UNCHANGED. The only per-record difference vs a diffusion sampler:
# step_map is [] (AR has no denoising steps) -> reward.py reports TPF = 0.
#
# Env: dllm-rl-ext (transformers 4.52.4, torch 2.6) runs Qwen2.5 natively —
# NO new env needed (unlike Dream which needs transformers 4.46.2).
#
# Pipeline (same driver shape as eval_dream_4bench_2node.sh):
#   math:  qwen_sample.py            ->                    aggregate_data.py -> reward.py
#   code:  qwen_sample.py(data_type=code) -> execute.py -> aggregate_data.py -> reward.py
# =====================================================================
import json, os, re, random
from jinja2 import Template
import torch
import multiprocessing as mp
from tqdm import tqdm
from termcolor import cprint
from transformers import AutoTokenizer, AutoModelForCausalLM

from omegaconf import OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


# --- ChatML prompt templates (identical to dream_sample.py; Qwen2.5-Instruct
#     uses exactly this <|im_start|>...<|im_end|> ChatML format) ----------------
system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''


def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


def get_token_lengths(strings, tokenizer):
    pad_token = tokenizer.pad_token
    escaped = re.escape(pad_token)
    pattern = rf"(?:{escaped})+"
    remove_pattern = escaped
    collapse_re = re.compile(pattern)
    lengths = []
    for s in strings:
        s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
        s_clean = re.sub(remove_pattern, '', s_clean)
        lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
    return lengths


def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, batch_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tok = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"     # decoder-only batched generation needs left pad

    try:
        model = (AutoModelForCausalLM.from_pretrained(
                    pretrained_model, trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2")
                 .to(device).eval())
    except Exception as e:
        cprint(f"[GPU {rank}] flash_attention_2 unavailable ({e}); falling back to sdpa", "yellow")
        model = (AutoModelForCausalLM.from_pretrained(
                    pretrained_model, trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa")
                 .to(device).eval())

    # stop on <|im_end|> (ChatML turn end) AND the tokenizer eos, if different
    eos_ids = []
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(imend, int) and imend >= 0:
        eos_ids.append(imend)
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)

    temperature = float(config.rollout.temperature)
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=int(config.rollout.max_gen_length),
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids if eos_ids else None,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = float(config.rollout.get("top_p", 1.0))
        tk = int(config.rollout.get("top_k", 0))
        if tk > 0:
            gen_kwargs["top_k"] = tk

    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # prompt string already contains full ChatML -> do NOT add special tokens
        enc = tok(batch_prompts, padding=True, return_tensors="pt",
                  add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        # keep ONLY the newly generated tokens (drop the left-padded prompt)
        gen_ids = out[:, enc["input_ids"].shape[1]:].tolist()
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
        for i, idx in enumerate(batch_idxs):
            seq_dict[idx]  = texts[i]
            step_dict[idx] = []      # AR: no diffusion step map (reward.py -> TPF 0)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    config = get_config()
    mp.set_start_method("spawn", force=True)

    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size

    project_name = config.experiment.project
    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)

    num = len(data)

    # build flat prompt list (each task repeated k_sample times)
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])

    cprint("start generation...", "green")

    # ---- shuffle -> even split across local GPUs -> launch one worker / GPU ----
    all_prompts = generation_prompts
    N = len(all_prompts)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "need >=1 GPU"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx,     n_gpu)

    manager   = mp.Manager()
    seq_dict  = manager.dict()
    step_dict = manager.dict()
    procs = []
    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                       args=(pretrained_model, rk,
                             prompt_chunks[rk], idx_chunks[rk],
                             seq_dict, step_dict, batch_size, config))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    restored_outputs   = [seq_dict[i]  for i in range(N)]
    restored_step_maps = [step_dict[i] for i in range(N)]
    cprint("generation job done!", "green")

    # ---- lengths + answer extraction + assemble per-record output ----
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    response_length = get_token_lengths(restored_outputs, tokenizer)

    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
