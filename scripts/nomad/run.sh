#!/bin/sh
# Runs inside the python:3.12-slim container on the cluster node. Plain POSIX sh.
# No dollar-brace, percent-brace or double-curly sequences (HCL heredoc and Nomad template rules).
# Steps: unpack the code, install the packages, wait for the course data (uploaded with nomad alloc exec,
# because the data is not public and the cluster has no shared storage), run the unit tests and
# python -m fedavg.gpu_checks, pack the results, then keep the allocation alive until they are fetched.
set -u
BASE=/alloc/data
WORK=$BASE/work
LOGS=$BASE/logs
OUT=$BASE/fedavg_gpu_results.tar.gz
mkdir -p $WORK $LOGS $BASE/in

pack() {
  cd $BASE
  if [ -d $WORK/results ]; then
    tar -czf $OUT.tmp -C $WORK results -C $BASE logs && mv $OUT.tmp $OUT
  else
    tar -czf $OUT.tmp -C $BASE logs && mv $OUT.tmp $OUT
  fi
  echo "PACKED $(wc -c < $OUT) bytes $(date -u +%H:%M:%S)"
  cd $WORK
}

hold() {
  # On this cluster Nomad removes a finished job and its files within hours, so the task stays alive until
  # /alloc/data/FETCHED exists (nomad alloc exec <alloc> touch /alloc/data/FETCHED) or HOLD_MINUTES have passed.
  echo "HOLD up to $HOLD_MINUTES min until /alloc/data/FETCHED exists"
  i=0
  while [ ! -e $BASE/FETCHED ] && [ $i -lt $HOLD_MINUTES ]; do
    sleep 60
    i=$(( i + 1 ))
  done
  echo "RELEASED after $i min $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

CORES=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))")
echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) node=$NODE_NAME cores=$CORES memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) shm=$(df -k /dev/shm | tail -n 1)"
base64 -d /local/code.b64 > $BASE/code.tar.gz
tar -xzf $BASE/code.tar.gz -C $WORK
cd $WORK
echo "pip $(date -u +%H:%M:%S)"
pip install --no-cache-dir --disable-pip-version-check -r requirements.txt > $LOGS/pip.log 2>&1
if [ $? -ne 0 ]; then
  echo "pinned install failed: $(tail -n 3 $LOGS/pip.log)"
  pip install --no-cache-dir --disable-pip-version-check torch ray numpy pandas scipy matplotlib pytest > $LOGS/pip_unpinned.log 2>&1 || { tail -n 20 $LOGS/pip_unpinned.log; pack; hold; exit 1; }
fi
python3 -c "import torch, ray, numpy; print('versions torch', torch.__version__, 'ray', ray.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

echo "WAITING_FOR_DATA $(date -u +%H:%M:%S)"
i=0
while [ ! -e $BASE/in/READY ] && [ $i -lt $DATA_WAIT_MINUTES ]; do
  sleep 10
  if [ -e $BASE/in/READY ]; then break; fi
  sleep 10
  if [ -e $BASE/in/READY ]; then break; fi
  sleep 10
  if [ -e $BASE/in/READY ]; then break; fi
  sleep 10
  if [ -e $BASE/in/READY ]; then break; fi
  sleep 10
  if [ -e $BASE/in/READY ]; then break; fi
  sleep 10
  i=$(( i + 1 ))
done
if [ ! -e $BASE/in/READY ]; then
  echo "NO_DATA after $i min"
  pack
  exit 1
fi
mkdir -p data
tar -xzf $BASE/in/data.tar.gz -C data
echo "DATA $(ls -la data | tr '\n' ' ')"

python3 -m pytest -q tests > $LOGS/pytest.log 2>&1
echo "pytest rc=$? $(tail -n 1 $LOGS/pytest.log)"

python3 -m fedavg.gpu_checks --out results/gpu > $LOGS/gpu_checks.log 2>&1 &
PID=$!
echo "started gpu_checks pid $PID $(date -u +%H:%M:%S)"
while kill -0 $PID 2>/dev/null; do
  sleep 60
  echo "$(date -u +%H:%M:%S) $(grep -v '^$' $LOGS/gpu_checks.log | tail -n 1 | cut -c1-200)"
  pack
done
wait $PID
echo "GPU_CHECKS rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
pack
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
hold
