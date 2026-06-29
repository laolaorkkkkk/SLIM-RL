import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
from collections import defaultdict

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import (
    get_config,
    flatten_omega_conf,
    AverageMeter,
    get_gdpo_config,
    generate_monotonic_pmasks_df,
)

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")






class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward, quadrature_weights=None, correctness=None):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward   = reward
        self.quadrature_weights = quadrature_weights
        self.correctness = correctness
        if p_mask.dim() == 3:
            logp_shape = (len(extended_input_ids), p_mask.shape[1], p_mask.shape[2])
        else:
            logp_shape = (len(extended_input_ids), p_mask.shape[1])
        self.logp_old_tok = torch.full(logp_shape, float("-inf"))

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
            self.quadrature_weights[idx] if self.quadrature_weights is not None else None,
            self.correctness[idx] if self.correctness is not None else True,
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_name

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_project = config.wandb.get("project", config.experiment.project)
        wandb_run_name = config.wandb.get("run_name", config.experiment.project)

        wandb_project = config.wandb.get("project", config.experiment.project)
        wandb_run_name = config.wandb.get("run_name", config.experiment.project)

        wandb_init_kwargs = dict(
            name=wandb_run_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            wandb_project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    #from transformers import AutoModelForCausalLM
    #model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # calculate loss ourselves, needs logits, so avoid fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False   
    

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")





    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward, quadrature_weights, correctness = zip(*batch)
        if len(batch) == 1 and extended_input_ids[0].dim() == 2:
            return {
                "ids": torch.tensor(idx),
                "extended_input_ids": extended_input_ids[0],
                "p_mask": p_mask[0],
                "tok_idx_ext": tok_idx_ext[0],
                "labels": labels[0],
                "reward": reward[0],
                "quadrature_weights": quadrature_weights[0],
                "correctness": torch.tensor(correctness, dtype=torch.bool),
            }
        return {
            "ids": torch.tensor(idx),
            "extended_input_ids": torch.stack(extended_input_ids),
            "p_mask": torch.stack(p_mask),
            "tok_idx_ext": torch.stack(tok_idx_ext),
            "labels": torch.stack(labels),
            "reward": reward,
            "quadrature_weights": torch.stack(quadrature_weights) if quadrature_weights[0] is not None else None,
            "correctness": torch.tensor(correctness, dtype=torch.bool),
        }
    


    
    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]

    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
    
    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))


    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num


    for x in dataset_load:
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

    
    
    def make_basic_block_attention(
        N: int,
        start_pos: int,            # = L0
        block_size: int,           # = b
    ) -> torch.Tensor:
        B = 1
        L0     = start_pos
        L1     = (N - L0) // 2          # N = L0 + 2·L1 
        assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

        # all -inf first
        bias = torch.full((B, 1, N, N), 0)


        rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
        rows_token = torch.arange(L0, L0 + L1)              # (L1,)

        # update block by block
        for bi in range((L1 + block_size - 1) // block_size):
            #  [bi*b , min((bi+1)*b, L1))
            left_end   = L0 + min((bi) * block_size, L1)        
            right_start= L0 + L1 + (left_end - L0)

            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)              # no i_end

            block_rows = rows[i_start:i_end]                    
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

            block_rows = rows_token[i_start:i_end]
            left_end   = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                # row interval [row_start, row_end)
                row_end   = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        
        return bias        # (B,1,N,N)


    
    
    

    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()



    def process_pad(attn, input_ids):
        N = input_ids.shape[1]
        device = input_ids.device

        cols = torch.arange(N, device=device)                  # (N,)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

        # set -inf
        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1  

        return attn






    




    def collect_training_data(input_ids, step_map_list, reward):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p

        labels_list = None
        quadrature_weights_list = None

        def _ensure_any_mask(pmask_tail, mask_id, noisy_seq, mask_offset):
            if pmask_tail.any():
                return pmask_tail, noisy_seq
            rand_pos = torch.randint(0, pmask_tail.numel(), (1,), device=pmask_tail.device).item()
            pmask_tail[rand_pos] = True
            noisy_seq[mask_offset + rand_pos] = mask_id
            return pmask_tail, noisy_seq

        
        if config.training.method == "slim_rl":
            n_points = config.training.get("num_quadrature_points", 3)
            gdpo_nodes, gdpo_weights = get_gdpo_config(n_points)

            extended_input_ids_list, pmask_list, reward_list = [], [], []
            quadrature_weights_list = []
            num_blocks = int((L1 - 1) / block_size) + 1

            for b in range(B):
                for t_node, w_node in zip(gdpo_nodes, gdpo_weights):
                    p_blocks = generate_monotonic_pmasks_df(
                        t_node=float(t_node),
                        num_blocks=num_blocks,
                        spread=config.training.get("mask_spread", 0.2),
                        shape=config.training.get("mask_shape", "cosine"),
                        device=input_ids.device,
                    )
                    if not config.training.get("mask_increasing", False):
                        p_blocks = p_blocks.flip(0)

                    extended_input_ids_b = input_ids[b]
                    pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                    for j in range(num_blocks):
                        start = j * block_size
                        end = min(L1, (j + 1) * block_size)

                        block_len = end - start
                        pmask_b_j = torch.rand(block_len, device=input_ids.device) <= p_blocks[j]
                        pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                        noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                        noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)
                        extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)

                    pmask_tail = pmask_b[L0:].clone()
                    extended_input_ids_b = extended_input_ids_b.clone()
                    pmask_tail, extended_input_ids_b = _ensure_any_mask(
                        pmask_tail, mask_id, extended_input_ids_b, L0 + L1
                    )
                    pmask_b[L0:] = pmask_tail

                    extended_input_ids_list.append(extended_input_ids_b)
                    pmask_list.append(pmask_b)
                    quadrature_weights_list.append(float(w_node))
                    reward_list.append(reward[b])

        else:
            raise ValueError(f"Unsupported training.method: {config.training.method}")

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
        pad_resp = (extended_input_ids[..., :L] == pad_id) & p_mask
        if post_num is not None:
            dim = 2 if pad_resp.dim() == 3 else 1
            cum_pad = torch.cumsum(pad_resp.int(), dim=dim)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        
        if labels_list is None:
            labels = extended_input_ids[..., :L].clone()
        else:
            labels = torch.stack(labels_list, dim=0)

        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
        tok_idx = valid.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]
        tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx = keep.nonzero(as_tuple=True)[0]

        extended_input_ids = extended_input_ids[idx]
        p_mask = p_mask[idx]
        tok_idx_ext = tok_idx_ext[idx]
        labels = labels[idx]

        reward_list = [reward_list[i] for i in idx.tolist()]

        if quadrature_weights_list is not None:
            quadrature_weights_list = [quadrature_weights_list[i] for i in idx.tolist()]

        if quadrature_weights_list is not None:
            quadrature_weights_list = torch.tensor(quadrature_weights_list, dtype=torch.float32)

        return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, quadrature_weights_list
        

    
    extended_input_ids, p_mask, tok_idx_ext, labels, rewards, quadrature_weights = collect_training_data(
        input_ids_lm, step_map_list, reward_list
    )

    correctness = torch.tensor([float(r) > 0 for r in rewards], dtype=torch.bool)

    dataset_lm = TrainDataset(
        extended_input_ids, p_mask, tok_idx_ext, labels, rewards, quadrature_weights, correctness=correctness
    )

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )





    

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )





    import torch.nn.functional as F


    @torch.no_grad()
    def compute_logp_old_tok_parallel(
            accelerator,
            dataset,
            train_dataloader_lm,
            start_pos, pad_id,
            batch_size):

        model.eval()

        dl = train_dataloader_lm

        for batch in dl:
            ids        = batch["ids"]         
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            is_3d = p_mask.dim() == 3
            if is_3d:
                B_orig, N_pts = p_mask.shape[:2]
                extended_input_ids = extended_input_ids.view(B_orig * N_pts, -1)
                p_mask = p_mask.view(B_orig * N_pts, -1)
                tok_idx_ext = tok_idx_ext.view(B_orig * N_pts, -1)
                labels = labels.view(B_orig * N_pts, -1)

            B, L = p_mask.shape
            L0    = start_pos
            L1    = L - L0
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
            logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            if is_3d:
                logp_tok = logp_tok.view(B_orig, N_pts, -1)

            if logp_tok.dim() == 2 and ids.numel() == 1 and dataset.logp_old_tok.dim() == 3:
                dataset.logp_old_tok[ids.item()] = logp_tok.float().cpu()
            else:
                dataset.logp_old_tok[ids] = logp_tok.float().cpu()

        accelerator.wait_for_everyone()

        model.train()


    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_logp_old_tok_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        batch_size=config.training.batch_size_lm,
    )






    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids_lm.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()
    global_step = 0
    metric_totals = defaultdict(float)
    metric_counts = defaultdict(float)
    
    # Epoch-level aggregators
    epoch_metric_totals = defaultdict(float)
    epoch_metric_counts = defaultdict(float)

    METRICS_NO_AVG = {"clip_count", "clip_total", "data/count"}

    


    

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok, quadrature_weights=None, correctness=None):

        adv = torch.as_tensor(
            adv, device=extended_input_ids.device, dtype=torch.float32
        ).detach()

        is_3d = p_mask.dim() == 3
        B_orig = 1
        if is_3d:
            B_orig, N_pts = p_mask.shape[:2]
            extended_input_ids = extended_input_ids.view(B_orig * N_pts, -1)
            p_mask = p_mask.view(B_orig * N_pts, -1)
            tok_idx_ext = tok_idx_ext.view(B_orig * N_pts, -1)
            labels = labels.view(B_orig * N_pts, -1)
            logp_old_tok = logp_old_tok.view(B_orig * N_pts, -1)
            adv = adv.unsqueeze(1).expand(B_orig, N_pts).reshape(B_orig * N_pts)
            if quadrature_weights is not None:
                quadrature_weights = quadrature_weights.view(B_orig * N_pts)

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
        logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

        log_probs = F.log_softmax(logits, dim=-1)
        
        logp_new_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)     # (B, T)

        if adv.numel() == 1 and B > 1:
            adv = adv.expand(B)

        imp_level = config.training.get("importance_sampling_level", "token")
        eps = config.training.eps

        if imp_level == "block":
            clip_ratio_low = config.training.get("clip_ratio_low", eps)
            clip_ratio_high = config.training.get("clip_ratio_high", eps)

            token_log_ratios = logp_new_tok - logp_old_tok
            token_log_ratios = torch.where(p_mask, token_log_ratios, torch.zeros_like(token_log_ratios))

            resp_log_ratios = token_log_ratios[:, L0:]    # (B, L1)
            resp_mask = p_mask[:, L0:]                     # (B, L1)

            block_size_imp = config.training.block_size
            K = (L1 + block_size_imp - 1) // block_size_imp
            pad_len = K * block_size_imp - L1

            resp_log_ratios_pad = F.pad(resp_log_ratios, (0, pad_len), value=0)
            resp_mask_pad = F.pad(resp_mask.float(), (0, pad_len), value=0)

            log_ratios_blk = resp_log_ratios_pad.view(B, K, block_size_imp)
            mask_blk = resp_mask_pad.view(B, K, block_size_imp)

            block_lengths = mask_blk.sum(dim=2)                        # (B, K)
            block_mask = block_lengths > 0                             # (B, K)
            block_lengths_clamped = block_lengths.clamp(min=1)

            mean_log_ratio_blk = (log_ratios_blk * mask_blk).sum(2) / block_lengths_clamped
            mean_log_ratio_blk = mean_log_ratio_blk.clamp(-10.0, 10.0)
            ratio_blk = torch.exp(mean_log_ratio_blk)                 # (B, K)
            clipped_blk = ratio_blk.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)

            adv_tok = adv.unsqueeze(1)                                 # (B, 1)
            surr_blk = torch.min(ratio_blk * adv_tok, clipped_blk * adv_tok) * block_mask  # (B, K)

            block_len_float = block_lengths * block_mask
            surrogate_tok = (surr_blk * block_len_float).sum(1) / L1  # (B,)

            clip_mask = ((ratio_blk < 1 - clip_ratio_low) | (ratio_blk > 1 + clip_ratio_high)) & block_mask
            clip_count = clip_mask.float().sum()
            clip_total = block_mask.float().sum()
        elif imp_level == "sequence":
            # Sequence-level IS (GSPO): one ratio per sequence = exp(mean masked-token log-ratio).
            # surrogate_tok here is the per-sequence POSITIVE surrogate min(ratio*adv, clip*adv) (shape (B,)),
            # matching the block/token branches; the downstream policy_loss = -(surrogate_tok*weights).sum()/B
            # negates it. No /L1 normalization: there is only ONE ratio per sequence.
            clip_ratio_low = config.training.get("clip_ratio_low", eps)
            clip_ratio_high = config.training.get("clip_ratio_high", eps)

            token_log_ratios = logp_new_tok - logp_old_tok
            token_log_ratios = torch.where(p_mask, token_log_ratios, torch.zeros_like(token_log_ratios))

            seq_lengths    = p_mask.float().sum(dim=1).clamp(min=1)            # (B,) masked-token count per sequence
            mean_log_ratio = token_log_ratios.sum(dim=1) / seq_lengths         # (B,) = log s_i(theta)
            mean_log_ratio = mean_log_ratio.clamp(-10.0, 10.0)
            ratio_seq   = torch.exp(mean_log_ratio)                            # (B,) s_i(theta)
            clipped_seq = ratio_seq.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)

            surrogate_tok = torch.min(ratio_seq * adv, clipped_seq * adv)      # (B,) per-sequence surrogate

            clip_mask  = (ratio_seq < 1 - clip_ratio_low) | (ratio_seq > 1 + clip_ratio_high)
            clip_count = clip_mask.float().sum()
            clip_total = torch.tensor(float(ratio_seq.numel()), device=ratio_seq.device, dtype=torch.float32)
        else:
            ratio   = logp_new_tok - logp_old_tok
            ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
            ratio   = torch.exp(ratio)          # (B, T)
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)            # (B, T)

            adv_tok = adv.unsqueeze(1)

            surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)  # (B, T)
            surrogate_tok = surrogate_tok * p_mask

            num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
            surrogate_tok = surrogate_tok.sum(dim=1) / L1

            clip_mask = ((ratio - clipped).abs() > 1e-8) & p_mask
            clip_count = clip_mask.float().sum()
            clip_total = p_mask.float().sum()

        if quadrature_weights is None:
            raise ValueError("quadrature_weights must be provided for slim_rl.")
        weights = quadrature_weights.to(device, dtype=surrogate_tok.dtype)
        policy_loss = - (surrogate_tok * weights).sum() / B

        # KL penalty (optional)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        kl_mean = torch.zeros((), device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok
            kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
            kl_estimator = config.training.get(
                "kl_estimator",
                "k3" if config.training.get("use_kl_estimator_k3", False) else "k1",
            )
            if kl_estimator == "k3":
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            elif kl_estimator == "k2":
                # Schulman's k2 second-order KL approximation.
                kl_seq = 0.5 * kl_seq.pow(2)
            kl_seq = (kl_seq * p_mask).sum(dim=1) / L1
            weights = quadrature_weights.to(device, dtype=kl_seq.dtype)
            kl_mean = (kl_seq * weights).sum() / B
            kl_loss = config.training.beta * (kl_seq * weights).sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss



        reward_mean = adv.mean()
        reward_std = adv.std(unbiased=False) if adv.numel() > 1 else torch.zeros((), device=adv.device)
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
        entropy_mean = (entropy * p_mask).sum() / p_mask.sum().clamp(min=1)
        mask_total = p_mask.float().sum()
        mask_ratio = mask_total / max(B * L1, 1)

        data_count = torch.tensor(float(B), device=extended_input_ids.device)

        metrics = {
            "loss/total": total_loss.detach(),
            "loss/policy": policy_loss.detach(),
            "loss/kl": kl_loss.detach(),
            "kl/mean": kl_mean.detach(),
            "policy/entropy": entropy_mean.detach(),
            "advantage/mean": reward_mean.detach(),
            "clip_count": clip_count.detach(),
            "clip_total": clip_total.detach(),
            "mask/ratio": mask_ratio.detach(),
            "mask/total": mask_total.detach(),
            "data/count": data_count.detach(),
        }

        return total_loss, metrics


    def calculate_metrics(totals, counts):
        result = {}
        for name, total in totals.items():
            if name in METRICS_NO_AVG:
                continue
            count = counts.get(name, 0.0)
            if count > 0:
                result[name] = total / count
        
        clip_total_total = totals.get("clip_total", 0.0)
        if clip_total_total > 0:
            clip_ratio = totals.get("clip_count", 0.0) / clip_total_total
            result["clip_ratio"] = clip_ratio
        if "data/count" in totals:
            result["data/count"] = totals["data/count"]
        return result

    def log_to_console(log_dict, step_label, step_val):
        if accelerator.is_main_process:
            print(f"[DEBUG] logging at {step_label}={step_val}, keys={list(log_dict.keys())}")
            for k, v in log_dict.items():
                try:
                    print(f"    {k} = {float(v):.6f}")
                except:
                    pass





    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        
        epoch_metric_totals.clear()
        epoch_metric_counts.clear()

        model.train()

        if accelerator.is_local_main_process:
            print("\n" + "=" * 120)
            print(f"[DEBUG] >>> ENTER EPOCH {epoch+1}/{num_train_epochs}")
            print(f"[DEBUG] dataset size = {len(dataset_lm)}, "
                  f"num_update_steps_per_epoch = {num_update_steps_per_epoch}")
            print("=" * 120 + "\n")
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,          
            leave=True               
        )
        
        

        for step, batch in enumerate(progress_bar, start=1):
            
            # for loss calculation

            data_time_m.update(time.time() - end)

            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            reward = batch["reward"]
            quadrature_weights = batch.get("quadrature_weights", None)
            correctness = batch["correctness"].to(accelerator.device)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)
            if old_lp.dim() == 3 and old_lp.size(0) == 1:
                old_lp = old_lp.squeeze(0)

            if torch.isneginf(old_lp).any().item():
                print(old_lp)

            loss_lm, step_metrics = forward_process(
                    extended_input_ids=extended_input_ids,
                    p_mask=p_mask,
                    tok_idx_ext=tok_idx_ext,
                    labels=labels,
                    adv=reward,
                    logp_old_tok=old_lp,
                    quadrature_weights=quadrature_weights,
                    correctness=correctness,
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps

            if accelerator.is_local_main_process and step <= 5:
                print(f"[DEBUG] step={step} (local) raw loss/total={step_metrics['loss/total'].item():.6f}, "
                      f"loss/policy={step_metrics['loss/policy'].item():.6f}, "
                      f"advantage/mean={step_metrics['advantage/mean'].item():.6f}")


            for name, value in step_metrics.items():
                gathered = accelerator.gather_for_metrics(value.detach())
                metric_totals[name] += gathered.sum().item()
                if name not in METRICS_NO_AVG:
                    metric_counts[name] += float(gathered.numel())


            if step < 10:
                print(loss_lm)
            accelerator.backward(loss_lm)

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                for name, total_val in metric_totals.items(): 
                    epoch_metric_totals[name] += total_val 
         
                for name, count_val in metric_counts.items(): 
                    epoch_metric_counts[name] += count_val 

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()
                
                global_step += 1

                log_dict = calculate_metrics(metric_totals, metric_counts)

                if lr_scheduler is not None:
                    log_dict["lr"] = lr_scheduler.get_last_lr()[0]

                if log_dict:
                    log_to_console(log_dict, "global_step", global_step)
                    # accelerator.log(log_dict, step=global_step)
                    if accelerator.is_main_process and "loss/total" in log_dict:
                        progress_bar.set_postfix({"loss": f"{log_dict['loss/total']:.4f}"})

                metric_totals.clear()
                metric_counts.clear()
                

        accelerator.wait_for_everyone()

        epoch_log_dict = calculate_metrics(epoch_metric_totals, epoch_metric_counts)
        


        # Log metrics 
        metrics_file_path = f"./{project_name}/temp_data/temp_metrics.json"
        external_metrics = {}

        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, 'r') as f:
                metrics_loaded = json.load(f)
            if metrics_loaded.get("epoch") == config.experiment.current_epoch:
                
                prefix = "train" if metrics_loaded.get("mode") == "train" else "eval"
                
                external_metrics = {
                    f"{prefix}/acc": metrics_loaded["acc"],
                    f"{prefix}/avg_length": metrics_loaded["avg_len"],
                    f"{prefix}/reward_mean": metrics_loaded.get("reward_mean", 0.0),
                    "advantage/prompts_total": metrics_loaded["prompts_total"],
                    "advantage/prompts_kept": metrics_loaded["prompts_kept"],
                    "advantage/prompts_dropped": metrics_loaded["prompts_dropped"],
                    "advantage/prompts_drop_rate": metrics_loaded["prompts_drop_rate"],
                    f"{prefix}/avg_unique_steps": metrics_loaded.get("avg_unique_steps", 0.0),
                    f"{prefix}/avg_tokens_per_step": metrics_loaded.get("avg_tokens_per_step", 0.0),
                    f"{prefix}/commit_count_mean": metrics_loaded.get("commit_count_mean", 0.0),
                    f"{prefix}/ewc_mean": metrics_loaded.get("ewc_mean", 0.0),
                    f"{prefix}/ewc_cumsum": metrics_loaded.get("ewc_cumsum", 0.0),
                }
                if accelerator.is_main_process:
                    logger.info(f"Loaded metrics from {metrics_file_path}: {external_metrics}")


        
      # Merge external metrics
        if external_metrics:
            epoch_log_dict.update(external_metrics)

        if accelerator.is_main_process:
            print("[DEBUG] epoch_log_dict:", {k: float(v) if isinstance(v, (int, float)) else v for k, v in epoch_log_dict.items()})
        if epoch_log_dict:
            epoch_step = config.experiment.current_epoch
            log_to_console(epoch_log_dict, "epoch", epoch_step)
            accelerator.log(epoch_log_dict, step=epoch_step)
            # --- persistent offline metrics log: append-only JSONL, one line per
            # epoch with the FULL metric set; survives wandb-server gaps and
            # per-epoch wandb-dir cleanup. Main process only, never breaks training. ---
            if accelerator.is_main_process:
                try:
                    import json as _json
                    _rec = {"step": int(epoch_step), "run_id": config.wandb.get("run_id", "")}
                    for _k, _v in epoch_log_dict.items():
                        _rec[_k] = (float(_v.item()) if hasattr(_v, "item")
                                    else (float(_v) if isinstance(_v, (int, float)) else _v))
                    _mh = Path(config.experiment.project) / f"metrics_history_{config.wandb.get('run_id', 'run')}.jsonl"
                    with open(_mh, "a") as _f:
                        _f.write(_json.dumps(_rec) + "\n")
                except Exception as _e:
                    logger.warning(f"metrics_history append failed: {_e}")

    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}")

    accelerator.end_training()





def save_checkpoint(model, tokenizer, config, accelerator, name):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)  # e.g. .../modeling_sdar.py
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)

                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")

    














if __name__ == "__main__":
    main()
