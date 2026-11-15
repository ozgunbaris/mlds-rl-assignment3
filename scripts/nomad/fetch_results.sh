#!/usr/bin/env bash
# Fetch the packed results of the GPU job and unpack them into results/gpu and results/gpu/cluster_logs.
# Usage (VPN on, SSH key loaded in ssh-agent, NOMAD_SSH=user@login-node): scripts/nomad/fetch_results.sh <alloc-id>
set -e
cd "$(dirname "$0")/../.."
ALLOC=${1:?alloc id}
: "${NOMAD_SSH:?set NOMAD_SSH to user@host of the Nomad login node}"
TMP=$(mktemp -d)
ssh -o ConnectTimeout=10 -o BatchMode=yes $NOMAD_SSH "nomad alloc fs $ALLOC alloc/data/fedavg_gpu_results.tar.gz" > $TMP/r.tar.gz
ls -la $TMP/r.tar.gz
tar -xzf $TMP/r.tar.gz -C $TMP
mkdir -p results/gpu/cluster_logs
[ -d $TMP/results/gpu ] && cp -R $TMP/results/gpu/. results/gpu/
cp -R $TMP/logs/. results/gpu/cluster_logs/
ssh -o ConnectTimeout=10 -o BatchMode=yes $NOMAD_SSH "nomad alloc logs $ALLOC run" > results/gpu/cluster_logs/job_stdout.log 2>/dev/null || true
rm -rf $TMP
ls results/gpu
echo FETCH_DONE
