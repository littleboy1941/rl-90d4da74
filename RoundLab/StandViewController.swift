import AVFoundation
import UIKit

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVSampleBufferDisplayLayer.self
    }

    var displayLayer: AVSampleBufferDisplayLayer {
        layer as! AVSampleBufferDisplayLayer
    }
}

// What the FX button cycles through: 3D face styles (ARKit) first,
// then flat Vision effects.
enum Mode {
    case face(FaceStyle)
    case flat(RoundEffect)

    var name: String {
        switch self {
        case .face(let style): return style.name
        case .flat(let effect): return effect.name
        }
    }

    // ARKit face styles need the front TrueDepth camera.
    var needsFront: Bool {
        if case .face = self {
            return true
        }
        return false
    }

    var title: String {
        Self.titles[name] ?? name
    }

    static let titles = [
        "blur": "Размытие лиц",
        "tv_static_body": "ТВ-шум на всё тело · быстро",
        "tv_static_body_hq": "ТВ-шум на всё тело · точно",
        "tv_static_body_40": "ТВ-шум на всё тело · как в 40",
        "tv_static": "ТВ-шум на лице",
        "scribble": "Зачёркнуто (маркер)",
        "smear": "Smear (стекание)",
        "ascii_robot": "ASCII · Mr. Robot",
        "ascii_lain": "ASCII · Lain",
        "acid": "Acid",
        "melt": "Melt (датамош)",
        "mr_swirl": "Mr. Swirl",
        "leak": "Вытекание",
        "text_face": "Лицо из текста",
        "ghost_mesh": "Призрак в сетке",
        "signal": "Сигнал",
        "black_silhouette": "Чёрный силуэт",
        "black_oval": "Чёрный овал",
    ]
}

final class StandViewController: UIViewController {
    private let output = RoundOutput()
    private lazy var visionSource = VisionSource(output: output, effect: allEffects[0])
    private lazy var faceSource: FaceSource? = FaceSource.isSupported
        ? FaceSource(output: output, style: allFaceStyles[0])
        : nil
    // The real effect (blur) first, then the 3D styles, then the rest.
    private lazy var modes: [Mode] = [Mode.flat(allEffects[0])]
        + (faceSource == nil ? [] : allFaceStyles.map { Mode.face($0) })
        + allEffects.dropFirst().map { Mode.flat($0) }
    private var backCamera = false
    private var debugBoxes = false
    private var active: FrameSource?

    private let preview = PreviewView()
    private let recordButton = UIButton(type: .system)
    private let effectButton = UIButton(type: .system)
    private let shareButton = UIButton(type: .system)
    private let cameraButton = UIButton(type: .system)
    private let statsLabel = UILabel()
    private let fileLabel = UILabel()
    private var modeIndex = 0

    private var frames = 0
    private var facesFrames = 0
    private var lostFrames = 0
    private var trackTotal = 0.0
    private var renderTotal = 0.0
    private var windowStart = CACurrentMediaTime()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        preview.displayLayer.videoGravity = .resizeAspectFill
        preview.backgroundColor = .darkGray
        preview.clipsToBounds = true
        preview.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(preview)

        statsLabel.textColor = .white
        statsLabel.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        statsLabel.numberOfLines = 0
        statsLabel.textAlignment = .center
        statsLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statsLabel)

        fileLabel.textColor = .lightGray
        fileLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        fileLabel.numberOfLines = 0
        fileLabel.textAlignment = .center
        fileLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(fileLabel)
        // Start on the effect being tested (falls back to the first mode).
        modeIndex = modes.firstIndex { $0.name == "tv_static_body" } ?? 0

        // Which TestFlight build is on the phone (CFBundleVersion = CI run number).
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        fileLabel.text = "build \(build)"

        recordButton.setTitle("REC", for: .normal)
        recordButton.setTitleColor(.white, for: .normal)
        recordButton.titleLabel?.font = .boldSystemFont(ofSize: 18)
        recordButton.backgroundColor = .systemRed
        recordButton.layer.cornerRadius = 36
        recordButton.translatesAutoresizingMaskIntoConstraints = false
        recordButton.addTarget(self, action: #selector(toggleRecording), for: .touchUpInside)
        view.addSubview(recordButton)

        effectButton.setTitle("FX", for: .normal)
        effectButton.setTitleColor(.white, for: .normal)
        effectButton.titleLabel?.font = .boldSystemFont(ofSize: 16)
        effectButton.backgroundColor = .darkGray
        effectButton.layer.cornerRadius = 28
        effectButton.translatesAutoresizingMaskIntoConstraints = false
        effectButton.showsMenuAsPrimaryAction = true
        view.addSubview(effectButton)

        cameraButton.setImage(UIImage(systemName: "arrow.triangle.2.circlepath.camera"), for: .normal)
        cameraButton.tintColor = .white
        cameraButton.backgroundColor = .darkGray
        cameraButton.layer.cornerRadius = 28
        cameraButton.translatesAutoresizingMaskIntoConstraints = false
        cameraButton.addTarget(self, action: #selector(flipCamera), for: .touchUpInside)
        view.addSubview(cameraButton)

        shareButton.setTitle("LOG", for: .normal)
        shareButton.setTitleColor(.white, for: .normal)
        shareButton.titleLabel?.font = .boldSystemFont(ofSize: 14)
        shareButton.backgroundColor = .darkGray
        shareButton.layer.cornerRadius = 28
        shareButton.translatesAutoresizingMaskIntoConstraints = false
        shareButton.addTarget(self, action: #selector(shareLogs), for: .touchUpInside)
        view.addSubview(shareButton)

        NSLayoutConstraint.activate([
            effectButton.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            effectButton.trailingAnchor.constraint(equalTo: recordButton.leadingAnchor, constant: -24),
            effectButton.widthAnchor.constraint(equalToConstant: 56),
            effectButton.heightAnchor.constraint(equalToConstant: 56),

            cameraButton.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            cameraButton.trailingAnchor.constraint(equalTo: effectButton.leadingAnchor, constant: -16),
            cameraButton.widthAnchor.constraint(equalToConstant: 56),
            cameraButton.heightAnchor.constraint(equalToConstant: 56),

            shareButton.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            shareButton.leadingAnchor.constraint(equalTo: recordButton.trailingAnchor, constant: 24),
            shareButton.widthAnchor.constraint(equalToConstant: 56),
            shareButton.heightAnchor.constraint(equalToConstant: 56),

            preview.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            preview.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -40),
            preview.widthAnchor.constraint(equalToConstant: 320),
            preview.heightAnchor.constraint(equalToConstant: 320),

            statsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            statsLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            statsLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            recordButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            recordButton.topAnchor.constraint(equalTo: preview.bottomAnchor, constant: 32),
            recordButton.widthAnchor.constraint(equalToConstant: 72),
            recordButton.heightAnchor.constraint(equalToConstant: 72),

            fileLabel.topAnchor.constraint(equalTo: recordButton.bottomAnchor, constant: 16),
            fileLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            fileLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
        ])

        output.onSample = { [weak self] sample, stats in
            self?.preview.displayLayer.sampleBufferRenderer.enqueue(sample)
            DispatchQueue.main.async {
                self?.collect(stats)
            }
        }

        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                self?.startCamera(granted: granted)
            }
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        preview.layer.cornerRadius = preview.bounds.width / 2
    }

    private func startCamera(granted: Bool) {
        guard granted else {
            statsLabel.text = "No camera access"
            return
        }
        apply(modes[modeIndex])
    }

    private func apply(_ mode: Mode) {
        effectButton.menu = makeMenu()
        let next: FrameSource?
        switch mode {
        case .face(let style):
            faceSource?.setStyle(style)
            next = faceSource
        case .flat(let effect):
            visionSource.setEffect(effect)
            next = visionSource
        }
        if active !== next {
            active?.stop()
            active = next
            active?.start()
        }
        statsLabel.text = mode.name
    }

    private func collect(_ stats: RoundOutput.Stats) {
        frames += 1
        trackTotal += stats.trackMs
        renderTotal += stats.renderMs
        if stats.faces > 0 {
            facesFrames += 1
        }
        if stats.lost {
            lostFrames += 1
        }
        let now = CACurrentMediaTime()
        let elapsed = now - windowStart
        guard elapsed >= 1 else {
            return
        }
        let count = Double(max(frames, 1))
        statsLabel.text = String(
            format: "%@\n%.1f fps / track %.1f ms / render %.1f ms\nface %d%% / lost %d%%",
            modes[modeIndex].name,
            Double(frames) / elapsed,
            trackTotal / count,
            renderTotal / count,
            facesFrames * 100 / max(frames, 1),
            lostFrames * 100 / max(frames, 1))
        frames = 0
        facesFrames = 0
        lostFrames = 0
        trackTotal = 0
        renderTotal = 0
        windowStart = now
    }

    // Effect list; on the back camera only the Vision effects are offered.
    private func makeMenu() -> UIMenu {
        let actions = modes.indices
            .filter { !backCamera || !modes[$0].needsFront }
            .map { index in
                UIAction(title: modes[index].title, state: index == modeIndex ? .on : .off) { [weak self] _ in
                    self?.select(index)
                }
            }
        let debug = UIAction(
            title: "Отладка: рамки лиц (Vision)",
            image: UIImage(systemName: "viewfinder"),
            state: debugBoxes ? .on : .off
        ) { [weak self] _ in
            self?.toggleDebug()
        }
        return UIMenu(title: backCamera ? "Задняя камера" : "Фронтальная камера", children: [
            UIMenu(options: .displayInline, children: actions),
            UIMenu(options: .displayInline, children: [debug]),
        ])
    }

    private func toggleDebug() {
        debugBoxes.toggle()
        visionSource.setDebug(debugBoxes)
        effectButton.menu = makeMenu()
    }

    private func select(_ index: Int) {
        modeIndex = index
        apply(modes[modeIndex])
    }

    // Back camera: Vision path only (ARKit face tracking is front-only);
    // a 3D style falls back to the blur.
    @objc private func flipCamera() {
        backCamera.toggle()
        visionSource.setPosition(backCamera ? .back : .front)
        if backCamera, modes[modeIndex].needsFront {
            modeIndex = 0
        }
        apply(modes[modeIndex])
    }

    // Newest CSV and newest MP4 from Documents, for sending without USB.
    @objc private func shareLogs() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: FrameLog.documents, includingPropertiesForKeys: nil)) ?? []
        let newest = ["csv", "mp4"].compactMap { ext in
            files.filter { $0.pathExtension == ext }
                .max { $0.lastPathComponent < $1.lastPathComponent }
        }
        guard !newest.isEmpty else {
            fileLabel.text = "no logs yet"
            return
        }
        let sheet = UIActivityViewController(activityItems: newest, applicationActivities: nil)
        sheet.popoverPresentationController?.sourceView = shareButton
        present(sheet, animated: true)
    }

    @objc private func toggleRecording() {
        if output.isRecording {
            recordButton.setTitle("REC", for: .normal)
            recordButton.backgroundColor = .systemRed
            output.stopRecording { [weak self] url in
                DispatchQueue.main.async {
                    self?.fileLabel.text = url.map { "saved: \($0.lastPathComponent)" } ?? "not saved"
                }
            }
        } else {
            recordButton.setTitle("STOP", for: .normal)
            recordButton.backgroundColor = .darkGray
            fileLabel.text = "recording..."
            output.startRecording()
        }
    }
}
