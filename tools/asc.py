"""App Store Connect API helper for RoundLab (key material is never printed).

    python tools/asc.py GET /v1/bundleIds "filter[identifier]=com.roundlab.stand"
    python tools/asc.py POST /v1/profiles body.json

Reads ISSUER_ID / KEY_ID / KEY_FILE from C:/Users/society/ios-signing/asc_api.txt.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import jwt

SIGNING = Path(r"C:\Users\society\ios-signing")
API = "https://api.appstoreconnect.apple.com"


def config():
    values = {}
    for line in (SIGNING / "asc_api.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Z_]+)\s*[:=]\s*(\S+)", line)
        if m:
            values[m.group(1)] = m.group(2)
    key_file = Path(values["KEY_FILE"])
    if not key_file.is_absolute():
        key_file = SIGNING / key_file.name
    return values["ISSUER_ID"], values["KEY_ID"], key_file.read_text(encoding="utf-8")


def token():
    issuer, key_id, key = config()
    now = int(time.time())
    return jwt.encode({"iss": issuer, "iat": now, "exp": now + 1000, "aud": "appstoreconnect-v1"},
                      key, algorithm="ES256", headers={"kid": key_id, "typ": "JWT"})


def call(method, path, query=None, body=None):
    url = API + path + ("?" + query if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            text = r.read().decode()
            return r.status, json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null")


def main():
    method, path = sys.argv[1], sys.argv[2]
    query = body = None
    for arg in sys.argv[3:]:
        if arg.endswith(".json"):
            body = json.loads(Path(arg).read_text(encoding="utf-8"))
        else:
            query = arg
    status, data = call(method, path, query, body)
    print(status)
    print(json.dumps(data, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
