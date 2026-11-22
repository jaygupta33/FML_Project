#!/usr/bin/env bash
# Copy project code into kaggle/code/ and create or version the private dataset jaygupta33/ns-surrogate-code.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/build_nsbench.py >/dev/null
rm -rf kaggle/code/src && mkdir -p kaggle/code/src
for d in nsbench optim solver; do [ -d "$d" ] && rsync -a --exclude __pycache__ "$d" kaggle/code/src/; done
cat > kaggle/code/dataset-metadata.json <<'J'
{"title": "ns-surrogate-code", "id": "jaygupta33/ns-surrogate-code", "isPrivate": true,
 "licenses": [{"name": "other"}]}
J
if kaggle datasets status jaygupta33/ns-surrogate-code >/dev/null 2>&1; then
  kaggle datasets version -p kaggle/code --dir-mode zip -m "${1:-update}"
else
  kaggle datasets create -p kaggle/code --dir-mode zip
fi
