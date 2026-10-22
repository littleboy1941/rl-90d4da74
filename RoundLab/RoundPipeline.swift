import AVFoundation
import CoreImage
import QuartzCore
import Vision

// A camera source that turns frames into 400x400 processed frames for RoundOutput.
protocol FrameSource: AnyObject {
    func start()
    func stop()
}

// Mirrors the Telegram round video path: front or back camera (32BGRA,
// up to 640x480) -> Vision face landmarks -> 2D effect -> 400x400 square.
// Preview and recording get the same processed frame, so nothing raw
// can reach the file.
final class VisionSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    var holdSeconds: CFTimeInterval = 0.5
    // Only a face found this many frames in a row is held after a miss, so a
    // one-frame false detection does not hang on screen for the hold time.
    var steadyFrames = 3
    // Weaker or smaller detections are dropped as false faces.
    var minConfidence: Float = 0.5
    var minFaceShare: CGFloat = 0.04          // of the frame width

    private let output: RoundOutput
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "roundlab.video")
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let sequenceHandler = VNSequenceRequestHandler()
    private var effect: RoundEffect
    private var configured = false
    private var lastPresentation: CMTime?
    private var lastFaces: [Face] = []
    private var lastFaceTime: CFTimeInterval = -1
    private var previousFaces: [Face] = []
    private var position: AVCaptureDevice.Position = .front
    private var input: AVCaptureDeviceInput?

    init(output: RoundOutput, effect: RoundEffect) {
        self.output = output
        self.effect = effect
    }

    func setEffect(_ next: RoundEffect) {
        queue.async { [self] in
            effect = next
        }
    }

    func start() {
        queue.async { [self] in
            if !configured {
                configured = (try? configure()) != nil
            }
            if !session.isRunning {
                session.startRunning()
            }
        }
    }

    func stop() {
        queue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    // Front <-> back. Held faces belong to the old camera, so drop them;
    // until the new camera finds a face the frame is fully covered.
    func setPosition(_ next: AVCaptureDevice.Position) {
        queue.async { [self] in
            guard next != position else {
                return
            }
            position = next
            lastFaces = []
            lastFaceTime = -1
            previousFaces = []
            guard configured else {
                return
            }
            session.beginConfiguration()
            defer { session.commitConfiguration() }
            try? attachCamera()
        }
    }

    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .vga640x480
        try attachCamera()
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
    }

    // Called inside a session configuration block.
    private func attachCamera() throws {
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: position
        ) else {
            throw NSError(domain: "RoundLab", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Camera not found",
            ])
        }
        if let input {
            session.removeInput(input)
        }
        let next = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(next) {
            session.addInput(next)
            input = next
        }
        // Buffers stay in the sensor orientation (landscape); the pipeline
        // rotates and mirrors them itself, like Telegram's round video filter.
        if (try? device.lockForConfiguration()) != nil {
            let frame = CMTime(value: 1, timescale: 30)
            let ranges = device.activeFormat.videoSupportedFrameRateRanges
            if ranges.contains(where: { $0.minFrameDuration <= frame && frame <= $0.maxFrameDuration }) {
                device.activeVideoMinFrameDuration = frame
                device.activeVideoMaxFrameDuration = frame
            }
            device.unlockForConfiguration()
        }
    }

    func captureOutput(
        _ captureOutput: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let presentation = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let intervalMs = lastPresentation.map { (presentation - $0).seconds * 1000 } ?? 0
        lastPresentation = presentation

        // Portrait: the sensor buffer is landscape; rotate it upright, and
        // mirror the front camera like a selfie preview.
        var source = CIImage(cvPixelBuffer: pixels)
        if source.extent.width > source.extent.height {
            source = source.oriented(position == .back ? .right : .leftMirrored)
        }
        source = source.transformed(by: CGAffineTransform(
            translationX: -source.extent.minX,
            y: -source.extent.minY))
        let width = Int(source.extent.width)
        let height = Int(source.extent.height)
        let size = CGSize(width: width, height: height)

        let visionStart = CACurrentMediaTime()
        var faces: [Face] = []
        var visionFailed = false
        let request = VNDetectFaceLandmarksRequest()
        do {
            try sequenceHandler.perform([request], on: source, orientation: .up)
            faces = (request.results ?? [])
                .filter { $0.confidence >= minConfidence }
                .map { observation in
                    Face(
                        rect: VNImageRectForNormalizedRect(observation.boundingBox, width, height),
                        contour: observation.landmarks?.faceContour?.pointsInImage(imageSize: size) ?? [])
                }
                .filter { $0.rect.width >= CGFloat(width) * minFaceShare }
            faces = faces.map { face in
                var face = face
                face.streak = 1 + (previousFaces
                    .filter { Self.overlap($0.rect, face.rect) > 0.3 }
                    .map(\.streak)
                    .max() ?? 0)
                return face
            }
        } catch {
            visionFailed = true
        }
        let visionEnd = CACurrentMediaTime()
        previousFaces = faces

        let steady = faces.filter { $0.streak >= steadyFrames }
        if !steady.isEmpty {
            lastFaces = steady
            lastFaceTime = visionEnd
        }
        let held = faces.isEmpty
            && lastFaceTime >= 0
            && visionEnd - lastFaceTime <= holdSeconds
        let holdShare = CGFloat((visionEnd - lastFaceTime) / holdSeconds)
        let used = faces.isEmpty
            ? (held ? lastFaces.map { var face = $0; face.hold = holdShare; return face } : [])
            : faces
        let lost = visionFailed || used.isEmpty

        let processed = effect.apply(to: source, faces: used, trackingLost: lost, time: presentation.seconds)
        let side = output.side
        let square = min(width, height)
        let crop = CGRect(
            x: (width - square) / 2,
            y: (height - square) / 2,
            width: square,
            height: square)
        let scale = CGFloat(side) / CGFloat(square)
        let framed = processed
            .cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        guard let target = output.makeBuffer() else {
            return
        }
        context.render(
            framed,
            to: target,
            bounds: CGRect(x: 0, y: 0, width: side, height: side),
            colorSpace: colorSpace)
        let renderEnd = CACurrentMediaTime()

        output.deliver(
            target,
            at: presentation,
            duration: CMSampleBufferGetDuration(sampleBuffer),
            stats: RoundOutput.Stats(
                intervalMs: intervalMs,
                trackMs: (visionEnd - visionStart) * 1000,
                renderMs: (renderEnd - visionEnd) * 1000,
                faces: faces.count,
                held: held,
                lost: lost))
    }

    // Intersection over union of two face boxes.
    static func overlap(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let common = a.intersection(b)
        guard !common.isNull else {
            return 0
        }
        let shared = common.width * common.height
        return shared / (a.width * a.height + b.width * b.height - shared)
    }
}
