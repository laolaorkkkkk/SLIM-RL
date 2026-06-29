#!/bin/bash
# Interactive multi-node RL launcher: run inside a 2-node salloc (not sbatch).
#   salloc --nodes=2 --ntasks-per-node=1 --gres=gpu:a100:4 --cpus-per-task=128 ...
#   bash scripts/train_inter.sh CONFIG [omegaconf overrides...]
# srun --overlaps the orchestrator onto the head compute node (the salloc shell
# is on the GPU-less login node); the worker node is dispatched via a nested
# srun --overlap (DLLM_REMOTE_LAUNCHER).

set -euo pipefail

# Require an allocation; node list comes from scontrol (robust on the login node).
JOBID="${SLURM_JOB_ID:?ERROR: not inside an allocation; run 'salloc ...' first, then bash scripts/train_inter.sh}"
CONFIG="${1:?ERROR: CONFIG required (never call bare, or it can collide with a running same-name run). e.g. bash scripts/train_inter.sh configs/multinode_slim_rl_sdar_align.yaml}"
shift || true                      # remaining "$@" = extra overrides
EXTRA_OVERRIDES=( "$@" )

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/path/to/venvs/dllm-rl/bin/activate}"          # training venv (edit or export VENV)
ORC_PY="${ORC_PY:-/path/to/venvs/orchestrator/bin/python}"   # host-side python with omegaconf for the orchestrator (edit or export ORC_PY)

NODELIST=$(scontrol show job "$JOBID" -o 2>/dev/null | grep -oP '(^| )NodeList=\K\S+' | head -1)
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
if [[ ${#NODES[@]} -lt 2 ]]; then
  echo "ERROR: this multinode config needs 2 nodes, but allocation $JOBID has ${#NODES[@]} (${NODES[*]:-none})." >&2
  echo "       Re-allocate with --nodes=2." >&2
  exit 1
fi
HEAD="${NODES[0]}"
WORKER="${NODES[1]}"
MASTER_PORT=$(( 20000 + JOBID % 40000 ))

# Topology vars the orchestrator reads.
export MLP_WORKER_NUM="${#NODES[@]}"
export MLP_WORKER_0_HOST="$HEAD"
export MLP_WORKER_1_HOST="$WORKER"
export MLP_WORKER_0_PORT="$MASTER_PORT"
export MASTER_ADDR="$HEAD"
export MASTER_PORT="$MASTER_PORT"

# Nested launcher for the worker node; --jobid pins it to THIS allocation.
export DLLM_REMOTE_LAUNCHER="srun --jobid=${JOBID} --overlap --cpu-bind=none --nodes=1 --ntasks=1 --nodelist={host} --cpus-per-task=${SLURM_CPUS_PER_TASK:-128} --gres=gpu:a100:4"
export DLLM_SKIP_INIT_HOSTS=1

# Per-node env prefix: module + venv + caches + tokens.
FAST_STORAGE="${FAST_STORAGE:-/path/to/fast_storage}"        # scratch dir for caches/tmp/output (edit or export FAST_STORAGE)
read -r -d '' DLLM_ENV_PREFIX <<EOF || true
module --force purge
module load pytorch/2.6
source ${VENV}
export HUGGINGFACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR="${FAST_STORAGE}/tmp"
export TEMP="${FAST_STORAGE}/tmp"
export TMP="${FAST_STORAGE}/tmp"
export XDG_CACHE_HOME="${FAST_STORAGE}/cache"
export HF_HOME="${FAST_STORAGE}/cache/hf_home"
export HF_MODULES_CACHE="${FAST_STORAGE}/cache/hf_home/modules"
export HF_DATASETS_CACHE="${FAST_STORAGE}/cache/hf_home/datasets"
export TRANSFORMERS_CACHE="${FAST_STORAGE}/cache/hf_home/hub"
export FLASHINFER_WORKSPACE_BASE="${FAST_STORAGE}/flashinfer_ws"
export FLASHINFER_CACHE_HOME="${FAST_STORAGE}/cache/flashinfer"
export CUDA_CACHE_PATH="${FAST_STORAGE}/cache/cuda"
export TORCH_HOME="${FAST_STORAGE}/cache/torch"
export TRITON_CACHE_DIR="${FAST_STORAGE}/tmp/.triton"
export WANDB_CACHE_DIR="${FAST_STORAGE}/tmp/.wandb_cache"
export WANDB_DIR="${FAST_STORAGE}/tmp/.wandb"
export WANDB_DATA_DIR="${FAST_STORAGE}/tmp/.wandb_data"
export OUTPUT_DIR="${FAST_STORAGE}/output"
export MASTER_ADDR="${HEAD}"
export MASTER_PORT="${MASTER_PORT}"
export MLP_WORKER_0_HOST="${HEAD}"
export MLP_WORKER_1_HOST="${WORKER}"
export MLP_WORKER_0_PORT="${MASTER_PORT}"
mkdir -p "${FAST_STORAGE}/tmp" "${FAST_STORAGE}/cache" "${FAST_STORAGE}/output" \
         "${FAST_STORAGE}/cache/hf_home" "${FAST_STORAGE}/cache/flashinfer" \
         "${FAST_STORAGE}/cache/cuda" "${FAST_STORAGE}/cache/torch" \
         "${FAST_STORAGE}/tmp/.triton" "${FAST_STORAGE}/tmp/.wandb_cache" "${FAST_STORAGE}/tmp/.wandb" 2>/dev/null || true
EOF
DLLM_ENV_PREFIX="$(printf '%s\n' "$DLLM_ENV_PREFIX" | grep -v '^[[:space:]]*$' \
    | awk 'NR==1{printf "%s",$0; next}{printf " && %s",$0}') && "
export DLLM_ENV_PREFIX

# Launch the orchestrator on the head compute node via srun --overlap. Do NOT
# module-load here: it must stay on the host ($ORC_PY) for its srun subprocess.
mkdir -p "${FAST_STORAGE}/tmp" "${FAST_STORAGE}/cache" "${FAST_STORAGE}/output" 2>/dev/null || true
LOGDIR="${LOGDIR:-${FAST_STORAGE}/logs}"
mkdir -p "$LOGDIR" 2>/dev/null || true
LOGFILE="${LOGDIR}/inter_${JOBID}.out"

echo "=================================================="
echo "INTERACTIVE run   Job: $JOBID   Nodes: ${NODES[*]}"
echo "Head(orch+node0): $HEAD"
echo "Worker(node1):    $WORKER"
echo "MasterPort:       $MASTER_PORT"
echo "Config:           $CONFIG"
echo "Overrides:        ${EXTRA_OVERRIDES[*]:-(none)}"
echo "RemoteLauncher:   $DLLM_REMOTE_LAUNCHER"
echo "Log (tee):        $LOGFILE"
echo "=================================================="

# line-buffer the stream so progress shows live through tee
srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD}" \
     --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" --gres=gpu:a100:4 \
     bash -c "cd '${PROJECT_ROOT}' && stdbuf -oL -eL '${ORC_PY}' multinode_rl.py config='${CONFIG}' ${EXTRA_OVERRIDES[*]}" \
  2>&1 | tee "$LOGFILE"
rc=${PIPESTATUS[0]}
echo "[train_inter] orchestrator exited rc=${rc}"
exit "$rc"
