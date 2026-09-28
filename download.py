"""Download the MIMIC-IV tables run.py needs (credentialed PhysioNet access).

  .venv/Scripts/python download.py [--version 3.1]

Credentials come from PHYSIONET_API_USERNAME / PHYSIONET_API_KEY (process env,
else the Windows user environment), falling back to a prompt.
Files already present are skipped;
partial downloads are written to *.part and only renamed when complete.
"""
import argparse
import getpass
import os
from pathlib import Path

import requests

FILES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/procedures_icd.csv.gz",
    "hosp/prescriptions.csv.gz",
    "icu/icustays.csv.gz",  # always loaded by PyHealth's MIMIC4EHRDataset
]

p = argparse.ArgumentParser()
p.add_argument("--version", default="3.1")
p.add_argument("--out", default="data/mimiciv")
a = p.parse_args()



def env(name):
    if name in os.environ:
        return os.environ[name]
    try:  # user env vars set after this shell started aren't inherited
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except (ImportError, OSError):
        return None


user = env("PHYSIONET_API_USERNAME") or input("PhysioNet username: ")
auth = (user, env("PHYSIONET_API_KEY") or getpass.getpass("PhysioNet password/key: "))
base = f"https://physionet.org/files/mimiciv/{a.version}"
root = Path(a.out) / a.version

with requests.Session() as s:
    s.auth = auth
    # PhysioNet answers 403 (no auth challenge) to the python-requests user agent
    s.headers["User-Agent"] = "Wget/1.21.4"
    for f in FILES:
        dest = root / f
        if dest.exists():
            print(f"skip {f}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        with s.get(f"{base}/{f}", stream=True, timeout=60) as r:
            if r.status_code in (401, 403):
                raise SystemExit(f"{r.status_code} on {f}: check credentials / MIMIC-IV access")
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r{f}: {done / total:6.1%} of {total / 1e6:.0f} MB", end="")
        part.rename(dest)
        print(f"\r{f}: done{' ' * 30}")

print(f"\nData in {root}. Run: .venv/Scripts/python run.py --task mortality --root {root.as_posix()}")
