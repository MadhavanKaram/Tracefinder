# tools/list_project_files.py
import os, json, hashlib
from pathlib import Path

ROOT = Path(".").resolve()
MAX_READ_BYTES = 1024  # for preview

def file_hash(path, algo="md5", max_bytes=65536):
    try:
        h = hashlib.new(algo)
        with open(path,"rb") as f:
            chunk = f.read(max_bytes)
            h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def preview_text(path, max_chars=400):
    try:
        if path.suffix.lower() in {".py",".md",".txt",".json",".yaml",".yml"}:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                t = f.read(max_chars)
            return t.replace("\r\n","\n")
    except Exception:
        pass
    return None

data = {"root": str(ROOT), "files": []}
for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        rel = p.relative_to(ROOT).as_posix()
        try:
            st = p.stat()
            entry = {
                "path": rel,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "hash": file_hash(p),
                "preview": preview_text(p)
            }
        except Exception as e:
            entry = {"path": rel, "error": str(e)}
        data["files"].append(entry)

# pretty print to stdout (so you can copy/paste)
print(json.dumps(data, indent=2))
