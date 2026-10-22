# RoundLab

Test stand for Telegram round video effects on iOS (target: the author's Telegram-iOS fork). Same path as a round video: camera -> effect -> round preview and 400x400 H.264 recording, the same processed frame for both.

- Sources: `FaceSource` (ARKit TrueDepth face mesh, SceneKit overlay, Metal passes) and `VisionSource` (AVCapture + Apple Vision landmarks, Core Image effects). The FX button cycles modes.
- Modes: `ascii_robot` / `ascii_lain` (head drawn with everyday Linux commands, one Metal pass), `blur` (real censoring: every face along its contour under an irreversible soft blur, both cameras), `acid` (psychedelic false color, contours, ripples, veins; melt feedback pass), `melt` (datamosh melt: block bands run down with feedback, one Metal pass), `mr_swirl` (Photoshop-style twirl on the face), `leak` (Elliot poster: LED matrix, face streaks, one Metal pass), `text_face`, `ghost_mesh`, `signal`, `black_silhouette`, `black_oval`.
- CI: `.github/workflows/build.yml` builds an unsigned IPA with `swiftc` (no Xcode project).
- Camera flip button (back camera: Vision effects only); FX opens the effect list.
- PC: `python tools/deploy.py` signs with zsign and installs over USB; `--pull` copies CSV logs and recordings to `logs/`.
- TestFlight: every push to main is also signed (Apple Distribution) and uploaded by CI when the signing secrets are set (`tools/set_secrets.py`); `python tools/wait_build.py <build>` waits for Apple processing. The LOG button shares the newest CSV and MP4.
- Logs: one CSV per launch in Documents: frame interval, track/render ms, faces, lost.
