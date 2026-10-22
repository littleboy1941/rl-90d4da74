"""Wait until the newest TestFlight build of RoundLab is processed.

    python tools/wait_build.py [build_number]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asc import call  # noqa: E402

APP = "6825664833"


def newest(number):
    query = (f"filter[app]={APP}&sort=-uploadedDate&limit=1&include=buildBetaDetail"
             "&fields[builds]=version,processingState,uploadedDate"
             "&fields[buildBetaDetails]=internalBuildState")
    if number:
        query += f"&filter[version]={number}"
    status, data = call("GET", "/v1/builds", query)
    if status != 200 or not data["data"]:
        return None
    build = data["data"][0]["attributes"]
    beta = next((i["attributes"] for i in data.get("included", [])), {})
    return build["version"], build["processingState"], beta.get("internalBuildState")


def main():
    number = sys.argv[1] if len(sys.argv) > 1 else None
    last = None
    for _ in range(60):
        state = newest(number)
        if state != last:
            print(time.strftime("%H:%M:%S"), state or "not visible yet", flush=True)
            last = state
        if state and state[1] in ("VALID", "INVALID", "FAILED") and state[2] != "PROCESSING":
            return
        time.sleep(60)
    print("timeout")


if __name__ == "__main__":
    main()
