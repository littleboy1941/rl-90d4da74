# RoundLab

Test stand for Telegram round video effects on iOS: front camera -> face tracking (Apple Vision) -> effect -> round preview and 400x400 H.264 recording, same frame for both.

- CI: `.github/workflows/build.yml` builds an unsigned IPA with `swiftc` (no Xcode project).
- PC: `python tools/deploy.py` signs with zsign and installs over USB; `--pull` copies CSV logs and recordings.
- Effects: `RoundLab/Effects.swift` (`RoundEffect` works on `CIImage`, like Telegram's round video filter).
