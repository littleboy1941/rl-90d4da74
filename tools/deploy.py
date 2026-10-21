"""RoundLab on the PC: download the last CI build, sign it, install over USB.

    python tools/deploy.py            # last successful build -> sign -> install
    python tools/deploy.py --pull     # copy Documents (CSV logs, MP4) to logs/

Signing material lives in C:/Users/society/ios-signing (never printed).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = r"C:\Program Files\GitHub CLI\gh.exe"
ZSIGN = r"C:\GamePort-work\tools\zsign\zsign.exe"
PMD = r"C:\Users\society\.local\bin\pymobiledevice3.exe"
SIGNING = Path(r"C:\Users\society\ios-signing")
P12 = SIGNING / "apple_dev.p12"
P12_PASSWORD = SIGNING / "p12_password.txt"
PROFILE = SIGNING / "RoundLab_Dev.mobileprovision"
BUNDLE = "com.roundlab.stand"


def run(args, **kwargs):
    print("$", " ".join(str(a) for a in args if not str(a).startswith("pw:")), flush=True)
    return subprocess.run([str(a).removeprefix("pw:") for a in args], check=True, **kwargs)


def last_run():
    out = subprocess.run(
        [GH, "run", "list", "--workflow", "build", "--limit", "1",
         "--json", "databaseId,status,conclusion,headSha,createdAt"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    runs = json.loads(out)
    if not runs:
        sys.exit("no CI runs")
    return runs[0]


def deploy():
    info = last_run()
    print("run", info)
    if info["status"] != "completed" or info["conclusion"] != "success":
        sys.exit("last CI run is not a finished success")
    folder = ROOT / "build" / f"run-{info['databaseId']}"
    ipa = folder / "RoundLab.ipa"
    if not ipa.exists():
        run([GH, "run", "download", str(info["databaseId"]), "-n", "RoundLab-ipa", "-D", folder], cwd=ROOT)
    if not PROFILE.exists():
        sys.exit(f"profile missing: {PROFILE}")
    signed = folder / "RoundLab_Signed.ipa"
    password = P12_PASSWORD.read_text(encoding="utf-8").strip()
    run([ZSIGN, "-k", P12, "-p", "pw:" + password, "-m", PROFILE, "-o", signed, ipa])
    run([PMD, "apps", "install", signed])
    print("installed", signed)


def pull():
    target = ROOT / "logs"
    target.mkdir(exist_ok=True)
    listing = subprocess.run(
        [PMD, "apps", "afc", "--documents", BUNDLE],
        input="ls /Documents\nexit\n", capture_output=True, text=True).stdout
    print(listing)
    for name in listing.split():
        if name.endswith((".csv", ".mp4")):
            run([PMD, "apps", "pull", "--documents", BUNDLE, f"/Documents/{name}", target / name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()
    pull() if args.pull else deploy()
