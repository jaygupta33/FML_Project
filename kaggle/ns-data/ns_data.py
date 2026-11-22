"""Download the nsforcing 128 data from Zenodo once; other kernels mount this kernel's output."""
import hashlib, os, tarfile, urllib.request, glob
import torch

URL = "https://zenodo.org/api/records/12825163/files/nsforcing_128.tgz/content"
MD5 = "70a389207ac93935d5ff3d4289d43581"
FILES = ("nsforcing_train_128.pt", "nsforcing_test_128.pt")
OUT = "/kaggle/working"

tgz, md5 = "/kaggle/tmp_nsforcing_128.tgz", hashlib.md5()
with urllib.request.urlopen(URL) as r, open(tgz, "wb") as fh:
    size, done, shown = int(r.headers.get("Content-Length") or 1450200928), 0, 0
    while chunk := r.read(1 << 22):
        fh.write(chunk); md5.update(chunk); done += len(chunk)
        if done * 10 // size > shown:
            shown = done * 10 // size
            print(f"[INFO] downloaded {done / 1e9:.2f} / {size / 1e9:.2f} GB", flush=True)
assert md5.hexdigest() == MD5, f"md5 mismatch: {md5.hexdigest()}"
with tarfile.open(tgz) as tar:
    tar.extractall(OUT, members=[m for m in tar.getmembers() if os.path.basename(m.name) in FILES], filter="data")
os.remove(tgz)
for f in FILES:
    hit = glob.glob(f"{OUT}/**/{f}", recursive=True)[0]
    if hit != f"{OUT}/{f}":
        os.replace(hit, f"{OUT}/{f}")
for f in FILES:
    d = torch.load(f"{OUT}/{f}")
    print(f"[INFO] {f}: " + ", ".join(f"{k} {tuple(v.shape)} std {v.float().std():.3f}" for k, v in d.items()))
print("[DONE] md5 verified, files in /kaggle/working")
