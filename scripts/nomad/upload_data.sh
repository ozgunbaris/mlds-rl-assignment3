#!/usr/bin/env bash
# Stream the course data into a running allocation of the GPU job and mark it ready.
# Usage (VPN on, SSH key loaded in ssh-agent, NOMAD_SSH=user@login-node): scripts/nomad/upload_data.sh <alloc-id>
set -e
cd "$(dirname "$0")/../.."
ALLOC=${1:?alloc id}
: "${NOMAD_SSH:?set NOMAD_SSH to user@host of the Nomad login node}"
SSH="ssh -o ConnectTimeout=10 -o BatchMode=yes $NOMAD_SSH"
TMP=$(mktemp -d)
COPYFILE_DISABLE=1 tar -czhf $TMP/data.tar.gz -C data train_data.npy test_data.npy
LOCAL=$(shasum -a 256 $TMP/data.tar.gz | cut -d' ' -f1)
ls -la $TMP/data.tar.gz
$SSH "nomad alloc exec -i -t=false $ALLOC sh -c 'cat > /alloc/data/in/data.tar.gz'" < $TMP/data.tar.gz
REMOTE=$($SSH "nomad alloc exec -t=false $ALLOC sha256sum /alloc/data/in/data.tar.gz" | cut -d' ' -f1)
echo "local  $LOCAL"
echo "remote $REMOTE"
[ "$LOCAL" = "$REMOTE" ] || { echo "CHECKSUM MISMATCH; READY not set"; exit 1; }
$SSH "nomad alloc exec -t=false $ALLOC touch /alloc/data/in/READY"
rm -rf $TMP
echo UPLOAD_DONE
