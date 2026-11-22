#!/usr/bin/env bash
set -euo pipefail

# Ensure local ChromaDB SQLite exists in ephemeral container disk (/tmp/chroma_db).
# Never run live ChromaDB SQLite directly over a GCS FUSE mount.
TARGET_DIR="${CHROMA_PERSIST_DIR:-/tmp/chroma_db}"
TARGET_DB="${TARGET_DIR}/chroma.sqlite3"
BUNDLED_SNAPSHOT="/app/data/chroma_db_snapshot.tar.gz"

if [ ! -f "${TARGET_DB}" ]; then
  mkdir -p "${TARGET_DIR}"

  if [ -n "${GCS_CHROMA_SNAPSHOT_URI:-}" ]; then
    echo "Downloading ChromaDB snapshot from ${GCS_CHROMA_SNAPSHOT_URI}..."
    python - <<'PY'
import json, os, tarfile, urllib.request
from pathlib import Path
from urllib.parse import quote

uri = os.environ["GCS_CHROMA_SNAPSHOT_URI"]
target = Path(os.environ.get("CHROMA_PERSIST_DIR", "/tmp/chroma_db"))
target.mkdir(parents=True, exist_ok=True)
tar_path = Path("/tmp/chroma_snapshot.tar.gz")

if uri.startswith("gs://"):
    _, rest = uri.split("gs://", 1)
    bucket, obj = rest.split("/", 1)
    meta_req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(meta_req, timeout=5) as r:
        token = json.loads(r.read().decode())["access_token"]
    dl_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(obj, safe='')}?alt=media"
    req = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as src, open(tar_path, "wb") as dst:
        dst.write(src.read())
else:
    urllib.request.urlretrieve(uri, tar_path)

with tarfile.open(tar_path, "r:gz") as tf:
    tf.extractall(target)
tar_path.unlink(missing_ok=True)
print(f"ChromaDB snapshot extracted from GCS to {target}")
PY
  elif [ -f "${BUNDLED_SNAPSHOT}" ]; then
    echo "Extracting bundled ChromaDB snapshot to ${TARGET_DIR}..."
    tar -xzf "${BUNDLED_SNAPSHOT}" -C "${TARGET_DIR}"
  fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
