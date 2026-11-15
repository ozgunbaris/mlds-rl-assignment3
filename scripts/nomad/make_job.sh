#!/usr/bin/env bash
# Generate scripts/nomad/$JOB_NAME.nomad: a Nomad batch job that runs the GPU checks of the Ray actors
# (python -m fedavg.gpu_checks) in a python:3.12-slim docker container on one node of the lab cluster.
# The code is embedded as a base64 tarball in a template block (the cluster has no shared storage).
# The course data is not embedded: the job waits until scripts/nomad/upload_data.sh has streamed it in.
# Defaults: mlds-deepdish5, 2 GPUs, 6 cores (4 actors, the driver, the Ray daemons), 16 GB of memory.
set -e
cd "$(dirname "$0")/../.."
REV=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
GPUS=${GPUS:-2}
CORES=${CORES:-6}
MEMORY=${MEMORY:-16384}
JOB_NAME=${JOB_NAME:-fedavg-gpu-checks}
NODE=${NODE:-mlds-deepdish5}
HOLD_MINUTES=${HOLD_MINUTES:-120}
DATA_WAIT_MINUTES=${DATA_WAIT_MINUTES:-30}
OUTFILE=scripts/nomad/$JOB_NAME.nomad
TMP=$(mktemp -d)
COPYFILE_DISABLE=1 tar -czf $TMP/code.tar.gz fedavg/*.py tests/*.py requirements.txt
B64=$(base64 < $TMP/code.tar.gz)
cat > $OUTFILE <<EOF2
job "$JOB_NAME" {
  datacenters = ["dc1"]
  type        = "batch"

  meta {
    revision = "$REV"
    gpus     = "$GPUS"
  }

  constraint {
    attribute = "\${attr.unique.hostname}"
    value     = "$NODE"
  }

  group "main" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    ephemeral_disk {
      sticky  = true
      migrate = false
      size    = 10000
    }

    task "run" {
      driver = "docker"

      config {
        image    = "python:3.12-slim"
        command  = "/bin/sh"
        args     = ["/local/run.sh"]
        shm_size = 2147483648
      }

      env {
        NODE_NAME         = "\${node.unique.name}"
        PYTHONUNBUFFERED  = "1"
        OMP_NUM_THREADS   = "1"
        MKL_NUM_THREADS   = "1"
        HOLD_MINUTES      = "$HOLD_MINUTES"
        DATA_WAIT_MINUTES = "$DATA_WAIT_MINUTES"
      }

      template {
        destination = "local/run.sh"
        data        = <<PAYLOAD_EOF
$(cat scripts/nomad/run.sh)
PAYLOAD_EOF
      }

      template {
        destination = "local/code.b64"
        data        = <<PAYLOAD_EOF
$B64
PAYLOAD_EOF
      }

      resources {
        cores  = $CORES
        memory = $MEMORY
        device "nvidia/gpu" {
          count = $GPUS
        }
      }
    }
  }
}
EOF2
rm -rf $TMP
echo "wrote $OUTFILE (job $JOB_NAME on $NODE, revision $REV, gpus $GPUS, cores $CORES, memory $MEMORY MB, hold $HOLD_MINUTES min)"
