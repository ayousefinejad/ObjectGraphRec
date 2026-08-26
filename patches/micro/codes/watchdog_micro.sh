#!/usr/bin/env bash
# Makes the 3-seed MICRO queue survive anything that kills the Claude session, including a
# process-group kill of the original queue (which nohup alone does not protect against).
#
# Launched detached from any session:
#   setsid nohup bash watchdog_micro.sh > /tmp/micro_watchdog.log 2>&1 < /dev/null &
#
# It does not start a second trainer while one is alive -- it only fills in seeds that are
# missing after the queue has stopped running, so it is safe to start while seed 1 is in flight.

set -u
cd "$(dirname "$0")"
PY=/home/worker/hamedenv/bin/python
export CUDA_VISIBLE_DEVICES=0
export LATTICE_EVAL_CORES=8

# A seed is complete when main.py has printed its final result dict.
done_seed () { grep -q "'precision': array" "/tmp/micro_seed$1.log" 2>/dev/null; }

# Anchored at the interpreter path on purpose. An unanchored "main\.py --dataset home_v2" also
# matches the Claude harness's own `bash -c ... eval '...pgrep -f "main.py --dataset home_v2"...'`
# wrapper processes, whose command lines contain that text verbatim -- which would make the
# watchdog believe a trainer was alive and sit idle forever after a real crash. Real trainers
# (and their eval-pool children, which inherit the cmdline) start with the venv python path;
# the bash wrappers start with /bin/bash.
trainer_alive () { pgrep -f "^/home/worker/hamedenv/bin/python main\.py --dataset home_v2" > /dev/null; }

echo "=== $(date '+%F %T')  watchdog up (pid $$)"

while true; do
  missing=()
  for S in 0 1 2; do done_seed "$S" || missing+=("$S"); done

  if [ ${#missing[@]} -eq 0 ]; then
    echo "=== $(date '+%F %T')  all three seeds complete"
    for S in 0 1 2; do
      printf 'seed %s: ' "$S"; grep -h "Test_Recall@20" "/tmp/micro_seed$S.log" | tail -1
    done
    echo "=== $(date '+%F %T')  micro watchdog done"
    exit 0
  fi

  if trainer_alive; then
    sleep 120
    continue
  fi

  # Nothing training and work outstanding -> the queue died. Take over the first missing seed.
  S=${missing[0]}
  echo "=== $(date '+%F %T')  queue not running, seeds left: ${missing[*]} -- restarting seed $S"
  $PY main.py --dataset home_v2 --gpu_id 0 --seed "$S" --epoch 400 > "/tmp/micro_seed$S.log" 2>&1
  echo "=== $(date '+%F %T')  seed $S exit=$?"
done
