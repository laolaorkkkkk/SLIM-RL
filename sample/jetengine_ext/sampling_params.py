from dataclasses import dataclass
from typing import Literal

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters
    block_length: int = 4
    denoising_steps: int = 4
    dynamic_threshold: float = 0.9
    eb_threshold: float = 0.35
    nlc_budget: float = 5.0
    tau_budget_m: float = 0  # 0 = auto (use num_to_transfer)
    topk: int = 0
    topp: float = 1
    remasking_strategy: Literal['sequential', 'low_confidence_static', 'low_confidence_dynamic', 'entropy_bounded', 'nlc_bounded_dynamic', 'contiguous_dynamic', 'gc_dynamic', 'tau_budget_dynamic'] = 'low_confidence_static'
    stop_words: list[int] | None = None
