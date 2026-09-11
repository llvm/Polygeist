#!/bin/bash
# run_jetson.sh — run a matched-and-cuBLAS-lowered kernel on a Jetson Orin.
#
# Three usage modes:
#
#   1. Named experiment (auto-runs the local e2e if needed):
#        ./scripts/correctness/run_jetson.sh <exp_name>
#      Currently supported names:  gemm
#      (Add more cases in the EXP→LOCAL_E2E table below as additional
#       kernels gain matcher coverage.)
#
#   2. Arbitrary post-Phase-2 MLIR file:
#        ./scripts/correctness/run_jetson.sh --mlir <path/to/abi.mlir> [tag]
#      The .mlir must already be ABI-lowered (i.e. has `func.call
#      @polygeist_*` ops, no residual `kernel.launch`). Optional tag
#      lets you label the staging dir on the Jetson; defaults to the file
#      basename minus .mlir.
#
#   3. Already-built Jetson/aarch64 executable:
#        ./scripts/correctness/run_jetson.sh --exe <path/to/exe> [tag]
#      The executable is staged and run directly. This is the most convenient
#      mode when `polygeist_build.sh --target=jetson` already produced a
#      binary locally and the Jetson does not have a Polygeist checkout.
#      Extra shared libraries can be staged beside it with:
#        POLYGEIST_JETSON_EXTRA_LIBS='/path/libA.so /path/libB.so' \
#          ./scripts/correctness/run_jetson.sh --exe ...
#
# Silicon profiles:
#
#   New PVA lab route:
#        POLYGEIST_SILICON_PROFILE=pva-general ./scripts/correctness/run_jetson.sh ...
#      This stages through arjaiswal@pva-general and then reaches Orin #2 at
#      nvidia@192.168.57.1. Its development toolkit is installed on the 1 TB
#      /home filesystem rather than the small root filesystem.
#
# Pipeline (this script glues together):
#   1. LOCAL  — run gemm_cublas_e2e.sh up through step (e) so that
#               /tmp/<exp>_cublas_test/<exp>_abi.mlir exists. That file
#               is the post-Phase-2 IR: kernel.launch ops have been
#               replaced by func.call to polygeist_cublas_dgemm.
#   2. SCP    — push <exp>_abi.mlir to the Jetson's staging dir.
#   3. REMOTE — on the Jetson, invoke build_jetson.sh to lower the .mlir
#               through one-shot-bufferize + LLVM dialect + mlir-translate
#               and link against /usr/local/cuda/lib64/libcublas.so. The
#               resulting binary calls real cuBLAS on the Orin GPU.
#   4. RUN    — execute the binary, capture stdout+stderr to a log.
#   5. SCP    — pull the log back to this VM's logs/ dir.
#
# Assumes Polygeist is already built on the Jetson at $JETSON_POLYGEIST
# (default: ~/Polygeist on the Jetson user's home). If not, run
# scripts/build_polygeist.sh on the Jetson once before using this script.
#
# Direct SSH from this VM → Jetson is the default (ssh-copy-id already done).
# If you need to bounce through a dev host, set ST_TRACKER_DEV_HOST or use a
# named POLYGEIST_SILICON_PROFILE, and the script will go DEV → JETSON like the
# exp_silicon.sh pattern.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <exp_name>           (named experiment, e.g. gemm)" >&2
    echo "       $0 --mlir <path> [tag]  (arbitrary post-ABI .mlir)" >&2
    echo "       $0 --exe <path> [tag]   (already-built Jetson executable)" >&2
    exit 1
fi

# Optional --dry-run: do all local prep + sanity checks, print what would
# happen on the Jetson, but don't actually scp / ssh anywhere.
DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

# Mode dispatch: either a named experiment, an explicit .mlir path, or an
# already-built executable.
MODE=""
EXP=""
LOCAL_MLIR=""
LOCAL_EXE=""

if [[ "$1" == "--mlir" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "ERROR: --mlir requires a path argument" >&2
    exit 1
  fi
  MODE="explicit"
  LOCAL_MLIR="$2"
  EXP="${3:-$(basename "$LOCAL_MLIR" .mlir)}"
  if [[ ! -s "$LOCAL_MLIR" ]]; then
    echo "ERROR: $LOCAL_MLIR does not exist or is empty" >&2
    exit 1
  fi
elif [[ "$1" == "--exe" || "$1" == "--binary" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "ERROR: --exe requires a path argument" >&2
    exit 1
  fi
  MODE="exe"
  LOCAL_EXE="$2"
  EXP="${3:-$(basename "$LOCAL_EXE")}"
  if [[ ! -s "$LOCAL_EXE" ]]; then
    echo "ERROR: $LOCAL_EXE does not exist or is empty" >&2
    exit 1
  fi
else
  MODE="named"
  EXP="$1"
fi

# ─── Configuration (override via env) ─────────────────────────────────────
JETSON_CREDENTIALS_FILE="${POLYGEIST_JETSON_CREDENTIALS_FILE:-${HOME}/.config/polygeist/jetson.env}"
if [[ -r "$JETSON_CREDENTIALS_FILE" ]]; then
  # Local, permission-restricted configuration; never commit credentials.
  # shellcheck disable=SC1090
  source "$JETSON_CREDENTIALS_FILE"
fi
SILICON_PROFILE="${POLYGEIST_SILICON_PROFILE:-${ST_TRACKER_PROFILE:-}}"
JETSON_HOST="${POLYGEIST_JETSON_HOST:-jetson-orin}"
JETSON_USER="${POLYGEIST_JETSON_USER:-nvidia}"
JETSON_STAGE="${POLYGEIST_JETSON_STAGE:-/tmp/polygeist_jetson_runs}"

# Dev-host bounce (mirrors exp_silicon.sh). With ST_TRACKER_DEV_HOST set,
# this VM key-auths into the dev box, which then uses sshpass with
# JETSON_PASS to reach the attached Jetson. When ST_TRACKER_DEV_HOST is unset,
# use direct SSH/SCP to POLYGEIST_JETSON_USER@POLYGEIST_JETSON_HOST.
DEV_HOST="${ST_TRACKER_DEV_HOST:-}"
DEV_USER="${ST_TRACKER_DEV_USER:-arjaiswal}"
DEV_DIR="${ST_TRACKER_DEV_DIR:-/colossus/workspace}"
JETSON_PASS="${ST_TRACKER_JETSON_PASS:-nvidia}"
JETSON_HOST_USB="${ST_TRACKER_JETSON_HOST:-192.168.57.1}"
JETSON_RUNS="${POLYGEIST_JETSON_RUNS:-1}"
JETSON_RUN_ARGS="${POLYGEIST_JETSON_RUN_ARGS:-}"
JETSON_EXTRA_LIBS="${POLYGEIST_JETSON_EXTRA_LIBS:-}"
JETSON_RT_TIMING="${POLYGEIST_RT_TIMING:-0}"
JETSON_RT_GRAPH_DIAGNOSTICS="${POLYGEIST_RT_GRAPH_DIAGNOSTICS:-0}"
JETSON_CUDA_GRAPH="${POLYGEIST_CUDA_GRAPH:-0}"
JETSON_GENERATED_GPU_DIAGNOSTICS="${POLYGEIST_GENERATED_GPU_DIAGNOSTICS:-0}"
JETSON_GENERATED_GPU_SYNCHRONOUS="${POLYGEIST_GENERATED_GPU_SYNCHRONOUS:-0}"

case "$SILICON_PROFILE" in
  ""|manual)
    ;;
  direct)
    DEV_HOST=""
    ;;
  pva-general)
    DEV_HOST="${ST_TRACKER_DEV_HOST:-pva-general}"
    DEV_USER="${ST_TRACKER_DEV_USER:-arjaiswal}"
    DEV_DIR="${ST_TRACKER_DEV_DIR:-/tmp}"
    JETSON_USER="${POLYGEIST_JETSON_USER:-nvidia}"
    JETSON_HOST_USB="${ST_TRACKER_JETSON_HOST:-192.168.57.1}"
    JETSON_PASS="${ST_TRACKER_JETSON_PASS:-nvidia}"
    ;;
  *)
    echo "ERROR: unknown POLYGEIST_SILICON_PROFILE='$SILICON_PROFILE'" >&2
    echo "       Supported profiles: pva-general, direct, manual" >&2
    exit 1
    ;;
esac

JETSON_POLYGEIST="${POLYGEIST_JETSON_DIR:-/home/${JETSON_USER}/Polygeist}"
JETSON_LD_LIBRARY_PATH="${POLYGEIST_JETSON_LD_LIBRARY_PATH:-/home/${JETSON_USER}/jetson-cuda-libs:/home/${JETSON_USER}/cuda-12.6/lib64:/home/${JETSON_USER}/cuda-12.6/targets/aarch64-linux/lib:/home/${JETSON_USER}/polygeist_cuda_libs:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:/home/${JETSON_USER}/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/${JETSON_USER}/cuda-12.6/targets/sbsa-linux/lib}"

# Where this script lives + where to drop logs locally.
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${POLYGEIST_LOG_DIR:-${SCRIPTS_DIR}/logs}"
mkdir -p "$LOG_DIR"

# ─── 1. LOCAL: ensure the ABI-lowered MLIR exists ─────────────────────────
if [[ "$MODE" == "named" ]]; then
  LOCAL_OUT="/tmp/${EXP}_cublas_test"
  LOCAL_MLIR="${LOCAL_OUT}/${EXP}_abi.mlir"

  case "$EXP" in
    gemm)
      LOCAL_E2E="${SCRIPTS_DIR}/gemm_cublas_e2e.sh"
      ;;
    *)
      echo "ERROR: unknown experiment '$EXP'. Add a case in $0 first," >&2
      echo "       or pass --mlir <path> with a pre-built ABI MLIR." >&2
      exit 1
      ;;
  esac

  if [[ ! -s "$LOCAL_MLIR" ]]; then
    mkdir -p "$LOCAL_OUT"
    echo "[$EXP] Local ABI-lowered MLIR not found; running ${LOCAL_E2E}..."
    # gemm_cublas_e2e.sh runs through step e by design and produces
    # ${LOCAL_OUT}/${EXP}_abi.mlir; the trailing CPU-stub build is harmless
    # for our purposes (we'll redo the build on the Jetson side anyway).
    bash "$LOCAL_E2E" >"${LOCAL_OUT}/local_e2e.log" 2>&1 || {
      echo "ERROR: local e2e failed. See ${LOCAL_OUT}/local_e2e.log" >&2
      tail -20 "${LOCAL_OUT}/local_e2e.log" >&2
      exit 1
    }
  fi

  if [[ ! -s "$LOCAL_MLIR" ]]; then
    echo "ERROR: $LOCAL_MLIR still missing after local e2e" >&2
    exit 1
  fi
fi

# Sanity-check that the .mlir is actually post-Phase-2 (has func.call to a
# polygeist_* shim and no residual kernel.launch). Executable mode has already
# been built and does not need this MLIR check.
N_LAUNCH=0
N_CALL=0
if [[ "$MODE" != "exe" ]]; then
  N_LAUNCH=$(grep -c '= kernel\.launch ' "$LOCAL_MLIR" 2>/dev/null || true)
  N_LAUNCH=${N_LAUNCH:-0}
  N_CALL=$(grep -cE 'call @polygeist_[A-Za-z0-9_]+' "$LOCAL_MLIR" 2>/dev/null || true)
  N_CALL=${N_CALL:-0}
  if [[ "$N_LAUNCH" != "0" ]] || [[ "$N_CALL" == "0" ]]; then
    echo "ERROR: $LOCAL_MLIR doesn't look ABI-lowered." >&2
    echo "       Found $N_LAUNCH residual kernel.launch and $N_CALL func.call to shim." >&2
    echo "       Run polygeist-opt --lower-kernel-launch-to-cublas on it first." >&2
    exit 1
  fi
fi

if [[ "$MODE" == "exe" ]]; then
  LOCAL_BYTES=$(stat -c '%s' "$LOCAL_EXE")
else
  LOCAL_BYTES=$(stat -c '%s' "$LOCAL_MLIR")
fi
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Experiment:    $EXP"
echo "  Profile:       ${SILICON_PROFILE:-manual}"
if [[ "$MODE" == "exe" ]]; then
  echo "  Local EXE:     $LOCAL_EXE ($LOCAL_BYTES bytes)"
else
  echo "  Local MLIR:    $LOCAL_MLIR ($LOCAL_BYTES bytes)"
fi
if [[ -n "$DEV_HOST" ]]; then
  echo "  Jetson:        ${JETSON_USER}@${JETSON_HOST_USB}"
else
  echo "  Jetson:        ${JETSON_USER}@${JETSON_HOST}"
fi
if [[ "$MODE" != "exe" ]]; then
  echo "  Polygeist on Jetson: $JETSON_POLYGEIST"
fi
if [[ -n "$DEV_HOST" ]]; then
  echo "  SSH mode:      bounce via ${DEV_USER}@${DEV_HOST}"
else
  echo "  SSH mode:      direct"
fi
echo "  Runs:          $JETSON_RUNS"
echo "═══════════════════════════════════════════════════════════════════════"

# ─── 2. SCP the .mlir to the Jetson ───────────────────────────────────────
STAGE_NAME="${EXP}_$(date +%Y%m%d_%H%M%S)"
REMOTE_DIR="${JETSON_STAGE}/${STAGE_NAME}"
LOG_NAME="${STAGE_NAME}.silicon.log"

# Helper: run a command on the Jetson via dev-host bounce + sshpass.
# Path: this VM --key--> dev box --sshpass+password--> Orin #2 (192.168.57.1).
# Matches the exp_silicon.sh access pattern.
jetson_ssh() {
  local cmd="$1"
  if [[ -n "$DEV_HOST" ]]; then
    printf '%s\n' "$cmd" | ssh -o BatchMode=yes "${DEV_USER}@${DEV_HOST}" \
      "sshpass -p '${JETSON_PASS}' ssh \
         -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o LogLevel=ERROR -o ConnectTimeout=10 \
         '${JETSON_USER}@${JETSON_HOST_USB}' bash -s"
  else
    printf '%s\n' "$cmd" | ssh -o BatchMode=yes -o ConnectTimeout=10 \
      "${JETSON_USER}@${JETSON_HOST}" bash -s
  fi
}

# Helper: scp a file to the Jetson. Two hops:
#   1. scp local file → dev box (key auth)
#   2. ssh into dev box, sshpass-scp from dev box → Jetson
jetson_scp_to() {
  local src="$1" dst="$2"
  local base
  base=$(basename "$src")
  if [[ -n "$DEV_HOST" ]]; then
    scp -q "$src" "${DEV_USER}@${DEV_HOST}:${DEV_DIR}/"
    ssh "${DEV_USER}@${DEV_HOST}" \
      "sshpass -p '${JETSON_PASS}' scp \
         -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o LogLevel=ERROR \
         '${DEV_DIR}/${base}' '${JETSON_USER}@${JETSON_HOST_USB}:${dst}'"
  else
    scp -q "$src" "${JETSON_USER}@${JETSON_HOST}:${dst}"
  fi
}

# Helper: scp a file FROM the Jetson back to this VM (two-hop reverse).
jetson_scp_from() {
  local src="$1" dst="$2"
  local base
  base=$(basename "$src")
  if [[ -n "$DEV_HOST" ]]; then
    ssh "${DEV_USER}@${DEV_HOST}" \
      "sshpass -p '${JETSON_PASS}' scp \
         -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o LogLevel=ERROR \
         '${JETSON_USER}@${JETSON_HOST_USB}:${src}' '${DEV_DIR}/${base}'"
    scp -q "${DEV_USER}@${DEV_HOST}:${DEV_DIR}/${base}" "$dst"
  else
    scp -q "${JETSON_USER}@${JETSON_HOST}:${src}" "$dst"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] Local prep OK."
  if [[ "$MODE" == "exe" ]]; then
    echo "[dry-run]   LOCAL_EXE=$LOCAL_EXE"
  else
    echo "[dry-run]   LOCAL_MLIR=$LOCAL_MLIR"
    echo "[dry-run]   N_LAUNCH=$N_LAUNCH residual ; N_CALL=$N_CALL polygeist_* call(s)"
  fi
  echo "[dry-run]   STAGE_NAME=$STAGE_NAME"
  echo "[dry-run]   REMOTE_DIR=$REMOTE_DIR"
  echo "[dry-run]   Log would land at ${LOG_DIR}/${LOG_NAME}"
  echo "[dry-run] Skipping scp/ssh; remove --dry-run to actually deploy."
  exit 0
fi

echo "[$EXP] Staging on Jetson: $REMOTE_DIR"
jetson_ssh "mkdir -p '$REMOTE_DIR'"
if [[ "$MODE" == "exe" ]]; then
  REMOTE_EXE="${REMOTE_DIR}/$(basename "$LOCAL_EXE")"
  jetson_scp_to "$LOCAL_EXE" "$REMOTE_EXE"
  if [[ -n "$JETSON_EXTRA_LIBS" ]]; then
    read -r -a EXTRA_LIB_FILES <<< "$JETSON_EXTRA_LIBS"
    for lib in "${EXTRA_LIB_FILES[@]}"; do
      if [[ ! -s "$lib" ]]; then
        echo "ERROR: extra runtime library missing or empty: $lib" >&2
        exit 1
      fi
      echo "[$EXP] Staging runtime library: $(basename "$lib")"
      jetson_scp_to "$lib" "${REMOTE_DIR}/$(basename "$lib")"
    done
  fi
else
  jetson_scp_to "$LOCAL_MLIR" "${REMOTE_DIR}/${EXP}_abi.mlir"
fi

# ─── 3 + 4. REMOTE: build on Jetson, run, capture log ────────────────────
if [[ "$MODE" == "exe" ]]; then
  EXE_PATH="$REMOTE_EXE"
else
  EXE_PATH="${REMOTE_DIR}/${EXP}_jetson_exe"
fi
REMOTE_LOG="${REMOTE_DIR}/${LOG_NAME}"

if [[ "$MODE" == "exe" ]]; then
  REMOTE_CMD="set -euo pipefail
  cd '$REMOTE_DIR'
  echo '[jetson] uname -a:' \$(uname -a)
  echo '[jetson] nvcc version:'
  nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null || echo '  (nvcc not found)'
  echo '[jetson] accelerator status:'
  if SMI_OUT=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null); then
    echo \"\$SMI_OUT\"
  elif command -v tegrastats >/dev/null 2>&1; then
    TS_LOG=/tmp/polygeist_tegrastats.\$\$
    tegrastats --interval 1000 --logfile \$TS_LOG >/dev/null 2>&1 &
    TS_PID=\$!
    sleep 1.2
    kill \$TS_PID 2>/dev/null || true
    wait \$TS_PID 2>/dev/null || true
    sed -n '1p' \$TS_LOG 2>/dev/null || true
    rm -f \$TS_LOG
  else
    echo '  (neither nvidia-smi nor tegrastats available)'
  fi
  echo
  chmod +x '$EXE_PATH'
  export LD_LIBRARY_PATH='$REMOTE_DIR:$JETSON_LD_LIBRARY_PATH':\${LD_LIBRARY_PATH:-}
  export POLYGEIST_RT_TIMING='$JETSON_RT_TIMING'
  export POLYGEIST_RT_GRAPH_DIAGNOSTICS='$JETSON_RT_GRAPH_DIAGNOSTICS'
  export POLYGEIST_CUDA_GRAPH='$JETSON_CUDA_GRAPH'
  if [ '$JETSON_GENERATED_GPU_DIAGNOSTICS' != 0 ]; then
    export POLYGEIST_GENERATED_GPU_DIAGNOSTICS=1
  fi
  if [ '$JETSON_GENERATED_GPU_SYNCHRONOUS' != 0 ]; then
    export POLYGEIST_GENERATED_GPU_SYNCHRONOUS=1
  fi
  TIMEFORMAT='[jetson] elapsed_s %3R'
  for i in \$(seq 1 '$JETSON_RUNS'); do
    echo '[jetson] running $EXE_PATH run' \$i '...'
    time '$EXE_PATH' $JETSON_RUN_ARGS 2>&1
  done
  echo
  echo '[jetson] exit code: 0'
"
else
  REMOTE_CMD="set -o pipefail
  cd '$REMOTE_DIR'
  echo '[jetson] uname -a:' \$(uname -a)
  echo '[jetson] nvcc version:'
  nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null || echo '  (nvcc not found)'
  echo '[jetson] accelerator status:'
  if SMI_OUT=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null); then
    echo \"\$SMI_OUT\"
  elif command -v tegrastats >/dev/null 2>&1; then
    TS_LOG=/tmp/polygeist_tegrastats.\$\$
    tegrastats --interval 1000 --logfile \$TS_LOG >/dev/null 2>&1 &
    TS_PID=\$!
    sleep 1.2
    kill \$TS_PID 2>/dev/null || true
    wait \$TS_PID 2>/dev/null || true
    sed -n '1p' \$TS_LOG 2>/dev/null || true
    rm -f \$TS_LOG
  else
    echo '  (neither nvidia-smi nor tegrastats available)'
  fi
  echo
  echo '[jetson] building binary via $JETSON_POLYGEIST/scripts/correctness/build_jetson.sh ...'
  bash '$JETSON_POLYGEIST/scripts/correctness/build_jetson.sh' \\
       '${REMOTE_DIR}/${EXP}_abi.mlir' '$EXE_PATH' 2>&1
  echo
  echo '[jetson] running $EXE_PATH ...'
  LD_LIBRARY_PATH='$JETSON_LD_LIBRARY_PATH':\${LD_LIBRARY_PATH:-} '$EXE_PATH' 2>&1
  echo
  echo '[jetson] exit code: '\$?
"
fi

echo "[$EXP] Running on Jetson (this may take a minute on first compile)..."
set +e
jetson_ssh "$REMOTE_CMD" 2>&1 | tee "${LOG_DIR}/${LOG_NAME}"
REMOTE_EXIT=${PIPESTATUS[0]}
set -e

# ─── 5. SCP the log back (mirror exp_silicon.sh) ──────────────────────────
# The local tee already captured the log, but pull a server-side copy too
# in case the network died mid-stream.
jetson_scp_from "$REMOTE_LOG" "${LOG_DIR}/${LOG_NAME}.remote" 2>/dev/null || true

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "Done. Local log: ${LOG_DIR}/${LOG_NAME}"
echo "Jetson exit code: $REMOTE_EXIT"
echo "═══════════════════════════════════════════════════════════════════════"
exit $REMOTE_EXIT
