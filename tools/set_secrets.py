"""Put the TestFlight signing secrets into the GitHub repo (values are never printed).

    python tools/set_secrets.py
"""
import base64
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asc import SIGNING, config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GH = r"C:\Program Files\GitHub CLI\gh.exe"


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def main():
    issuer, key_id, key = config()
    secrets = {
        "DIST_P12_BASE64": b64(SIGNING / "distribution.p12"),
        "DIST_P12_PASSWORD": (SIGNING / "dist_p12_password.txt").read_text(encoding="utf-8").strip(),
        "PROFILE_APP_BASE64": b64(SIGNING / "RoundLab_AppStore.mobileprovision"),
        "ASC_KEY_P8_BASE64": base64.b64encode(key.encode()).decode(),
        "ASC_KEY_ID": key_id,
        "ASC_ISSUER_ID": issuer,
    }
    for name, value in secrets.items():
        subprocess.run([GH, "secret", "set", name], input=value, text=True, cwd=ROOT,
                       check=True, capture_output=True)
        print("set", name)


if __name__ == "__main__":
    main()
