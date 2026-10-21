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

final class StandViewController: UIViewController {
    private let pipeline = RoundPipeline()
    private let preview = PreviewView()
    private let recordButton = UIButton(type: .system)
    private let effectButton = UIButton(type: .system)
    private let statsLabel = UILabel()
    private let fileLabel = UILabel()
    private var effectIndex = 0

    private var frames = 0
    private var facesFrames = 0
    private var lostFrames = 0
    private var visionTotal = 0.0
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
        effectButton.addTarget(self, action: #selector(nextEffect), for: .touchUpInside)
        view.addSubview(effectButton)

        NSLayoutConstraint.activate([
            effectButton.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            effectButton.trailingAnchor.constraint(equalTo: recordButton.leadingAnchor, constant: -32),
            effectButton.widthAnchor.constraint(equalToConstant: 56),
            effectButton.heightAnchor.constraint(equalToConstant: 56),

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

        pipeline.onFrame = { [weak self] sample, stats in
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
        do {
            try pipeline.configure()
            pipeline.start()
            statsLabel.text = allEffects[effectIndex].name
        } catch {
            statsLabel.text = "Camera error: \(error.localizedDescription)"
        }
    }

    private func collect(_ stats: RoundPipeline.Stats) {
        frames += 1
        visionTotal += stats.visionMs
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
        statsLabel.text = String(
            format: "%@\n%.1f fps · vision %.1f ms\nface %d%% · black %d%%",
            allEffects[effectIndex].name,
            Double(frames) / elapsed,
            visionTotal / Double(max(frames, 1)),
            facesFrames * 100 / max(frames, 1),
            lostFrames * 100 / max(frames, 1))
        frames = 0
        facesFrames = 0
        lostFrames = 0
        visionTotal = 0
        windowStart = now
    }

    @objc private func nextEffect() {
        effectIndex = (effectIndex + 1) % allEffects.count
        pipeline.setEffect(allEffects[effectIndex])
        statsLabel.text = allEffects[effectIndex].name
    }

    @objc private func toggleRecording() {
        if pipeline.isRecording {
            recordButton.setTitle("REC", for: .normal)
            recordButton.backgroundColor = .systemRed
            pipeline.stopRecording { [weak self] url in
                DispatchQueue.main.async {
                    self?.fileLabel.text = url.map { "saved: \($0.lastPathComponent)" } ?? "not saved"
                }
            }
        } else {
            recordButton.setTitle("STOP", for: .normal)
            recordButton.backgroundColor = .darkGray
            fileLabel.text = "recording…"
            pipeline.startRecording()
        }
    }
}
