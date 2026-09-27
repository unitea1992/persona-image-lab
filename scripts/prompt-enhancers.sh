#!/usr/bin/env bash
set -euo pipefail
umask 077

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
runtime="$root/cache/prompt-enhancer"
mkdir -p -- "$runtime"
chmod 700 -- "$runtime"

python_bin="${PERSONA_VLLM_PYTHON:-python3}"
ninja_bin="$(command -v ninja 2>/dev/null || true)"
if [[ -z "$ninja_bin" && -x "$HOME/.local/bin/ninja" ]]; then
  ninja_bin="$HOME/.local/bin/ninja"
fi

have_vllm() {
  command -v "$python_bin" >/dev/null 2>&1 &&
    "$python_bin" -c 'import vllm' >/dev/null 2>&1 &&
    [[ -n "$ninja_bin" ]]
}

pid_file() { printf '%s/%s.pid' "$runtime" "$1"; }
socket_file() { printf '%s/%s.sock' "$runtime" "$1"; }
log_file() { printf '%s/%s.log' "$runtime" "$1"; }

alive() {
  local file pid cmdline expected_socket expected_model
  file="$(pid_file "$1")"
  [[ -s "$file" ]] || return 1
  pid="$(cat "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  expected_socket="$(socket_file "$1")"
  expected_model="persona-pe-$1"
  grep -Fq 'vllm.entrypoints.openai.api_server' <<<"$cmdline" &&
    grep -Fq -- "--uds $expected_socket" <<<"$cmdline" &&
    grep -Fq -- "--served-model-name $expected_model" <<<"$cmdline"
}

ready() {
  local socket
  alive "$1" || return 1
  socket="$(socket_file "$1")"
  [[ -S "$socket" ]] || return 1
  chmod 600 -- "$socket" 2>/dev/null || return 1
  command -v curl >/dev/null 2>&1 || return 1
  curl --fail --silent --show-error --max-time 2 \
    --unix-socket "$socket" http://localhost/health >/dev/null 2>&1
}

start_one() {
  local name="$1" model_dir="$2" model_name="$3" max_len="$4"
  local pidfile socket logfile cache_dir
  pidfile="$(pid_file "$name")"
  socket="$(socket_file "$name")"
  logfile="$(log_file "$name")"
  cache_dir="$runtime/$name-cache"

  if alive "$name"; then
    printf '%s already running (pid %s).\n' "$name" "$(cat "$pidfile")"
    return 0
  fi

  rm -f -- "$pidfile" "$socket"
  mkdir -p -- "$cache_dir/torch"
  PATH="$(dirname -- "$ninja_bin"):$PATH" \
  VLLM_CACHE_ROOT="$cache_dir" \
  TORCHINDUCTOR_CACHE_DIR="$cache_dir/torch" \
  nohup "$python_bin" -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --uds "$socket" \
    --dtype bfloat16 \
    --quantization fp8_per_tensor \
    --max-model-len "$max_len" \
    --gpu-memory-utilization 0.50 \
    --kv-cache-memory-bytes 1G \
    --max-num-seqs 1 \
    --served-model-name "$model_name" \
    --enforce-eager \
    >"$logfile" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$pidfile"
  printf 'Starting %s (pid %s). Log: %s\n' "$name" "$!" "$logfile"
}

stop_one() {
  local name="$1" pidfile socket pid
  pidfile="$(pid_file "$name")"
  socket="$(socket_file "$name")"
  if alive "$name"; then
    pid="$(cat "$pidfile")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f -- "$pidfile" "$socket"
}

status_one() {
  local name="$1" socket
  socket="$(socket_file "$name")"
  if alive "$name"; then
    if ready "$name"; then
      printf '%s: ready (pid %s)\n' "$name" "$(cat "$(pid_file "$name")")"
    else
      printf '%s: loading (pid %s)\n' "$name" "$(cat "$(pid_file "$name")")"
    fi
  else
    printf '%s: stopped\n' "$name"
  fi
}

case "${1:-status}" in
  start)
    have_vllm || {
      printf 'Host vLLM and ninja are required for the accelerated prompt enhancer backend.\n' >&2
      exit 2
    }
    start_one t2i "$root/model/prompt-enhancer-t2i" persona-pe-t2i 4096
    start_one i2i "$root/model/prompt-enhancer-i2i" persona-pe-i2i 8192
    ;;
  stop)
    stop_one t2i
    stop_one i2i
    ;;
  status)
    status_one t2i
    status_one i2i
    ;;
  *)
    printf 'Usage: %s {start|stop|status}\n' "$0" >&2
    exit 2
    ;;
esac
