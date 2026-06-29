# SLIM-RL: Risk-Budgeted Random-Masking RL for Diffusion LLMs Without Trajectory Slicing


## 🚀 Quick Start

Install into a fresh virtualenv:

```bash
python3.12 -m venv dllm-rl
source dllm-rl/bin/activate
pip install --upgrade pip
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -r requirements.txt
```

## ⚙️ Data

```bash
cd data
python download_data.py --dataset MATH500
python download_data.py --dataset MATH_train
cd ..
```

## 📊 Inference & Evaluations

Evaluate across benchmarks (MATH500 / GSM8K / MBPP / HumanEval) × decoding strategies × block sizes in one launch on a 2-node SLURM allocation. Edit the paths and `MODEL` at the top of the script first:

```bash
SLURM_JOB_ID=<2-node allocation> bash scripts/eval_sdar4b_wotau_4bench_2node.sh
```

## 🔧 Reinforcement Learning

On a SLURM cluster, `scripts/train_inter.sh` wraps the multi-node launch and it starts the per-node `srun` workers around `multinode_rl.py` (set your Hugging Face / W&B tokens at the top of the script first):

```bash
bash scripts/train_inter.sh configs/multinode_slim_rl_sdar.yaml
```
