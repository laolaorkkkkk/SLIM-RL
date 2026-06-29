#!/bin/bash
# 2-node SDAR-4B 4-bench eval (len=256), each run sharded across both nodes.
#   benches:    MATH500, GSM8K (math) ; MBPP, HumanEval (code)
#   strategies: low_confidence_dynamic (lcd) ; tau_budget_dynamic (taubudget)
#   block_size: 16, 4   ->  4 benches x 2 strats x 2 bs = 16 runs
#   n per bench: MATH500=9 GSM8K=3 MBPP=9 HumanEval=9
# math path: sdar_sample.py -> aggregate_data.py -> reward.py  (model=<ckpt>)
# code path: sdar_rl_rollout.py -> rl_execute.py -> rl_aggregate_data.py -> rl_code_reward.py
#            (model.pretrained_model=<ckpt> + experiment.current_epoch=1)
# Run from the login node with the 2-node alloc job id:
#   SLURM_JOB_ID=<jobid> bash scripts/eval_sdar4b_wotau_4bench_2node.sh                     # all 16
#   SLURM_JOB_ID=<jobid> bash scripts/eval_sdar4b_wotau_4bench_2node.sh MBPP_taubudget_bs16 # one
# Env knobs: STRATS, BSIZES, BENCHES, MODEL, PREFIX, NREP_OVERRIDE.
set -uo pipefail

JOBID="${SLURM_JOB_ID:?ERROR: run with the 2-node alloc job id, e.g. SLURM_JOB_ID=<jobid> bash scripts/eval_sdar4b_wotau_4bench_2node.sh}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/path/to/venvs/dllm-rl/bin/activate}"   # eval venv (edit or export VENV)
FAST="${FAST:-/path/to/fast_storage}"                 # scratch dir for caches/tmp (edit or export FAST)
MODEL="${MODEL:-/path/to/SDAR-4B-Chat}"               # loadable model dir or ckpt (edit or export MODEL)
PREFIX="${PREFIX:-eval_sdar4b}"

# --- bench -> data_type ; per-bench n ; strategy tag -> remasking_strategy ---
declare -A DTYPE=(   [MATH500]=math [GSM8K]=math [MBPP]=code [HumanEval]=code )
declare -A NREP=(    [MATH500]=9    [GSM8K]=3    [MBPP]=9    [HumanEval]=9 )
declare -A STRATOF=( [lcd]=low_confidence_dynamic [taubudget]=tau_budget_dynamic )
read -ra BENCHES <<< "${BENCHES:-MATH500 GSM8K MBPP HumanEval}"   # env override: BENCHES="MBPP HumanEval" -> code-only re-run
read -ra STRATS  <<< "${STRATS:-lcd taubudget}"   # env override allowed
read -ra BSIZES  <<< "${BSIZES:-16 4}"
NREP_OVERRIDE="${NREP_OVERRIDE:-}"   # env override: force per-task n for ALL benches this run (e.g. NREP_OVERRIDE=20); empty = per-bench defaults above

# --- build run matrix: TAG = <BENCH>_<lcd|taubudget>_bs<BS> ---
ALL_TAGS=()
for B in "${BENCHES[@]}"; do for S in "${STRATS[@]}"; do for BS in "${BSIZES[@]}"; do ALL_TAGS+=("${B}_${S}_bs${BS}"); done; done; done
RUN_TAGS=( "$@" ); [ ${#RUN_TAGS[@]} -eq 0 ] && RUN_TAGS=( "${ALL_TAGS[@]}" )

# --- authoritative 2-node list (anchor on NodeList= to avoid ReqNodeList=(null)) ---
NODELIST=$(scontrol show job "$JOBID" -o 2>/dev/null | grep -oP '(^| )NodeList=\K\S+' | head -1)
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
[ ${#NODES[@]} -lt 2 ] && { echo "ERROR: need 2 nodes, alloc $JOBID has ${#NODES[@]} (${NODES[*]:-none})." >&2; exit 1; }
N0="${NODES[0]}"; N1="${NODES[1]}"

[ -d "$MODEL" ] || { echo "ERROR: model dir not found: $MODEL" >&2; exit 1; }

read -r -d '' ENVP <<EOF || true
module --force purge && module load pytorch/2.6 && source ${VENV} \
 && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 && export TMPDIR=${FAST}/tmp TEMP=${FAST}/tmp TMP=${FAST}/tmp \
 && export XDG_CACHE_HOME=${FAST}/cache HF_HOME=${FAST}/cache/hf_home \
 && export TRANSFORMERS_CACHE=${FAST}/cache/hf_home/hub \
 && export TRITON_CACHE_DIR=${FAST}/tmp/.triton CUDA_CACHE_PATH=${FAST}/cache/cuda \
 && export FLASHINFER_WORKSPACE_BASE=${FAST}/flashinfer_ws FLASHINFER_CACHE_HOME=${FAST}/cache/flashinfer \
 && export NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0 DS_SKIP_CUDA_CHECK=1 \
 && mkdir -p ${FAST}/tmp ${FAST}/cache/hf_home ${FAST}/cache/cuda ${FAST}/tmp/.triton ${FAST}/cache/flashinfer ${FAST}/flashinfer_ws
EOF

srun_on () {  # srun_on <node> <bash-command>
  srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --nodelist="$1" \
       --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" --gres=gpu:a100:4 \
       bash -c "${ENVP} && $2"
}

# slowest-worker decode time = max ELAPSED across every tqdm bar in the per-node logs.
pbar_decode_of () {
  ( export LC_ALL=C
    mx=0
    for f in "$@"; do
      [ -f "$f" ] || continue
      while read -r tok; do
        [ -z "$tok" ] && continue
        s=$(echo "$tok" | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }')
        [ "${s:-0}" -gt "$mx" ] 2>/dev/null && mx=$s
      done < <(tr '\r' '\n' < "$f" 2>/dev/null | grep -aoE '\[[0-9]+:[0-9:]+<' | tr -d '[<')
    done
    if [ "$mx" -gt 0 ] 2>/dev/null; then printf "%d:%02d" $((mx/60)) $((mx%60)); else printf "n/a"; fi )
}

echo "=================================================="
echo "SDAR-4B wo_tau (ep140) 4-bench eval  len256  Job:$JOBID   Nodes: $N0(idx0)+$N1(idx1)"
echo "Model: $MODEL"
echo "Runs (${#RUN_TAGS[@]}): ${RUN_TAGS[*]}"
echo "=================================================="

run_one () {  # run_one <BENCH> <lcd|taubudget> <BS>
  local B="$1" S="$2" BS="$3"
  local DT="${DTYPE[$B]}" STRAT="${STRATOF[$S]}" N="${NREP[$B]}"; [ -n "$NREP_OVERRIDE" ] && N="$NREP_OVERRIDE"
  local PROJ="${PREFIX}_${B}_${S}_bs${BS}"

  mkdir -p "${ROOT}/${PROJ}/results" "${ROOT}/${PROJ}/temp_data" 2>/dev/null || true
  rm -f "${ROOT}/${PROJ}/results"/results-eval-*.txt 2>/dev/null || true   # reward appends; clear stale
  local LOG0="${ROOT}/${PROJ}/results/decode-n0.log" LOG1="${ROOT}/${PROJ}/results/decode-n1.log"
  rm -f "$LOG0" "$LOG1" 2>/dev/null || true

  echo; echo ">>> [${PROJ}] sample (sharded $N0+$N1)  $(date '+%H:%M:%S')   ${STRAT} ${DT} bs${BS} n=${N}"
  local P0 P1 R0 R1
  if [ "$DT" = "math" ]; then
    # math path: model=<ckpt> (top-level string), rollout.* overrides
    local CFG="../configs/eval/eval_sdar4b_${B}_dynamic.yaml"
    local OV="experiment.project=${PROJ} experiment.num_node=2 model=${MODEL} dataset.eval_dataset=${B} \
rollout.block_size=${BS} rollout.denoising_steps_per_block=${BS} rollout.remasking_strategy=${STRAT} \
rollout.num_response_per_task=${N} rollout.top_k=0"
    [ "$S" = "taubudget" ] && OV="${OV} rollout.tau_budget_m=1"

    srun_on "$N0" "cd '${ROOT}/sample' && stdbuf -oL -eL python sdar_sample.py config=${CFG} ${OV} experiment.node_index=0" 2> >(tee "$LOG0" >&2) & P0=$!
    srun_on "$N1" "cd '${ROOT}/sample' && stdbuf -oL -eL python sdar_sample.py config=${CFG} ${OV} experiment.node_index=1" 2> >(tee "$LOG1" >&2) & P1=$!
    wait "$P0"; R0=$?; wait "$P1"; R1=$?
    [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ] && { echo "!! [${PROJ}] sample failed (rc $R0/$R1) -> skip"; return 1; }
    sleep 0.5; local DECODE; DECODE="$(pbar_decode_of "$LOG0" "$LOG1")"
    echo ">>> [${PROJ}] gen-decode (slowest worker): ${DECODE}"

    echo ">>> [${PROJ}] aggregate (node0)"
    srun_on "$N0" "cd '${ROOT}/reward' && python aggregate_data.py config=${CFG} ${OV}" || { echo "!! [${PROJ}] aggregate failed"; return 1; }
    echo ">>> [${PROJ}] reward/score (node0)"
    srun_on "$N0" "cd '${ROOT}/reward' && python reward.py config=${CFG} ${OV}" || { echo "!! [${PROJ}] reward failed"; return 1; }
  else
    # code path: model.pretrained_model=<ckpt> + current_epoch=1, evaluation.* overrides
    local CFG="../configs/eval_sdar4b_code_base.yaml"
    # eval_dataset + data_type go in COMMON so every substep agrees (else execute/
    # aggregate fall back to the cfg default and miss the rollout's output file).
    local COMMON="config=${CFG} experiment.current_epoch=1 experiment.function=evaluation experiment.num_node=2 \
experiment.project=${PROJ} model.pretrained_model=${MODEL} evaluation.eval_dataset=${B} evaluation.data_type=code"
    local EVALARGS="evaluation.remasking_strategy=${STRAT} \
evaluation.block_size=${BS} evaluation.denoising_steps_per_block=${BS} evaluation.num_response_per_task=${N} evaluation.top_k=0"
    [ "$S" = "taubudget" ] && EVALARGS="${EVALARGS} evaluation.tau_budget_m=1"

    srun_on "$N0" "cd '${ROOT}/sample' && stdbuf -oL -eL python sdar_rl_rollout.py ${COMMON} ${EVALARGS} experiment.node_index=0" 2> >(tee "$LOG0" >&2) & P0=$!
    srun_on "$N1" "cd '${ROOT}/sample' && stdbuf -oL -eL python sdar_rl_rollout.py ${COMMON} ${EVALARGS} experiment.node_index=1" 2> >(tee "$LOG1" >&2) & P1=$!
    wait "$P0"; R0=$?; wait "$P1"; R1=$?
    [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ] && { echo "!! [${PROJ}] sample failed (rc $R0/$R1) -> skip"; return 1; }
    sleep 0.5; local DECODE; DECODE="$(pbar_decode_of "$LOG0" "$LOG1")"
    echo ">>> [${PROJ}] gen-decode (slowest worker): ${DECODE}"

    echo ">>> [${PROJ}] execute (sharded $N0+$N1)"
    srun_on "$N0" "cd '${ROOT}/reward' && stdbuf -oL -eL python rl_execute.py ${COMMON} experiment.node_index=0" & local X0=$!
    srun_on "$N1" "cd '${ROOT}/reward' && stdbuf -oL -eL python rl_execute.py ${COMMON} experiment.node_index=1" & local X1=$!
    wait "$X0"; local XR0=$?; wait "$X1"; local XR1=$?
    [ "$XR0" -ne 0 ] || [ "$XR1" -ne 0 ] && { echo "!! [${PROJ}] execute failed (rc $XR0/$XR1) -> skip"; return 1; }

    echo ">>> [${PROJ}] aggregate (node0)"
    srun_on "$N0" "cd '${ROOT}/reward' && python rl_aggregate_data.py ${COMMON}" || { echo "!! [${PROJ}] aggregate failed"; return 1; }
    echo ">>> [${PROJ}] reward/score (node0)  -> 'train step: 1'"
    srun_on "$N0" "cd '${ROOT}/reward' && python rl_code_reward.py ${COMMON} ${EVALARGS}" || { echo "!! [${PROJ}] reward failed"; return 1; }
  fi

  # persist slowest-worker decode time into the result file (standing rule: every eval task saves it)
  local rf; rf="$(ls "${ROOT}/${PROJ}/results"/results-eval-*.txt 2>/dev/null | head -1)"
  [ -n "$rf" ] && echo "gen-decode (slowest worker): ${DECODE}" >> "$rf"
  echo "    -> $(grep -hE 'acc:|train step: 1 |gen-decode' "${ROOT}/${PROJ}/results"/results-eval-*.txt 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' || echo 'NO RESULT ROW')"
}

for TAG in "${RUN_TAGS[@]}"; do
  B="${TAG%%_*}"; rest="${TAG#*_}"; S="${rest%%_*}"; BS="${rest##*_bs}"
  case " ${BENCHES[*]} " in *" $B "*) : ;; *) echo "!! unknown bench in tag '$TAG' -> skip"; continue;; esac
  run_one "$B" "$S" "$BS"
done

echo; echo "===== SUMMARY  (model: ${MODEL}  |  prefix: ${PREFIX}  |  4-bench len256) ====="
for TAG in "${ALL_TAGS[@]}"; do
  printf "%-30s %s\n" "$TAG" "$(grep -hE 'acc:|train step: 1 |gen-decode' "${ROOT}/${PREFIX}_${TAG}/results"/results-eval-*.txt 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' || echo '<not run>')"
done
echo "==================================================================="
