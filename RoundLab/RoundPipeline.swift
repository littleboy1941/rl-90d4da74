import AVFoundation
import CoreImage
import QuartzCore
import Vision

// Mirrors the Telegram round video path: front camera (32BGRA, up to
// 640x480) -> effect -> 400x400 square -> preview and H.264 writer.
// Preview and recording get the same processed frame, so nothing raw
// can reach the file.
final class RoundPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    struct Stats {
        var intervalMs: Double
        var visionMs: Double
        var renderMs: Double
        var faces: Int
        var held: Bool
        var lost: Bool
    }

    let session = AVCaptureSession()
    let side = 400
    var effect: RoundEffect = allEffects[0]
    var holdSeconds: CFTimeInterval = 0.5

    // Called on the video queue.
    var onFrame: ((CMSampleBuffer, Stats) -> Void)?

    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "roundlab.video")
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let sequenceHandler = VNSequenceRequestHandler()
    private var pool: CVPixelBufferPool?
    private var lastPresentation: CMTime?
    private var lastFaces: [Face] = []
    private var lastFaceTime: CFTimeInterval = -1
    private var writer: RoundWriter?
    private let log = FrameLog()

    func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .vga640x480
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            throw NSError(domain: "RoundLab", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Front camera not found",
            ])
        }
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) {
            session.addOutput(output)
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

    func start() {
        queue.async { [session] in
            if !session.isRunning {
                session.startRunning()
            }
        }
    }

    func setEffect(_ next: RoundEffect) {
        queue.async { [self] in
            effect = next
        }
    }

    var isRecording: Bool {
        queue.sync { writer != nil }
    }

    func startRecording() {
        queue.async { [self] in
            guard writer == nil else {
                return
            }
            writer = try? RoundWriter(side: side)
        }
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        queue.async { [self] in
            guard let current = writer else {
                completion(nil)
                return
            }
            writer = nil
            current.finish(completion: completion)
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let presentation = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let intervalMs = lastPresentation.map { (presentation - $0).seconds * 1000 } ?? 0
        lastPresentation = presentation

        // Front camera, portrait: the sensor buffer is landscape; rotate it
        // upright and mirror it like a selfie preview.
        var source = CIImage(cvPixelBuffer: pixels)
        if source.extent.width > source.extent.height {
            source = source.oriented(.leftMirrored)
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
            faces = (request.results ?? []).map { observation in
                Face(
                    rect: VNImageRectForNormalizedRect(observation.boundingBox, width, height),
                    contour: observation.landmarks?.faceContour?.pointsInImage(imageSize: size) ?? [])
            }
        } catch {
            visionFailed = true
        }
        let visionEnd = CACurrentMediaTime()

        if !faces.isEmpty {
            lastFaces = faces
            lastFaceTime = visionEnd
        }
        let held = faces.isEmpty
            && lastFaceTime >= 0
            && visionEnd - lastFaceTime <= holdSeconds
        let used = faces.isEmpty ? (held ? lastFaces : []) : faces
        let lost = visionFailed || used.isEmpty

        let processed = effect.apply(to: source, faces: used, trackingLost: lost, time: presentation.seconds)
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

        guard let target = makeBuffer() else {
            return
        }
        context.render(
            framed,
            to: target,
            bounds: CGRect(x: 0, y: 0, width: side, height: side),
            colorSpace: colorSpace)
        let renderEnd = CACurrentMediaTime()

        writer?.append(target, at: presentation)

        let stats = Stats(
            intervalMs: intervalMs,
            visionMs: (visionEnd - visionStart) * 1000,
            renderMs: (renderEnd - visionEnd) * 1000,
            faces: faces.count,
            held: held,
            lost: lost)
        log.write(presentation: presentation, stats: stats, recording: writer != nil)

        if let sample = makeSample(target, like: sampleBuffer, at: presentation) {
            onFrame?(sample, stats)
        }
    }

    private func makeBuffer() -> CVPixelBuffer? {
        if pool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: side,
                kCVPixelBufferHeightKey as String: side,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            ]
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        }
        guard let pool else {
            return nil
        }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }

    private func makeSample(
        _ buffer: CVPixelBuffer,
        like original: CMSampleBuffer,
        at presentation: CMTime
    ) -> CMSampleBuffer? {
        var format: CMVideoFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: nil,
            imageBuffer: buffer,
            formatDescriptionOut: &format)
        guard let format else {
            return nil
        }
        var timing = CMSampleTimingInfo(
            duration: CMSampleBufferGetDuration(original),
            presentationTimeStamp: presentation,
            decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: nil,
            imageBuffer: buffer,
            formatDescription: format,
            sampleTiming: &timing,
            sampleBufferOut: &sample)
        if let sample,
           let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: true),
           CFArrayGetCount(attachments) > 0 {
            let dictionary = unsafeBitCast(
                CFArrayGetValueAtIndex(attachments, 0),
                to: CFMutableDictionary.self)
            CFDictionarySetValue(
                dictionary,
                Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
        }
        return sample
    }
}

// H.264 400x400 MP4 in Documents, like a Telegram round video (no audio yet).
final class RoundWriter {
    let url: URL
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private var started = false

    init(side: Int) throws {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        url = FrameLog.documents.appendingPathComponent("round-\(formatter.string(from: Date())).mp4")
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: side,
            AVVideoHeightKey: side,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 1_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ])
        input.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        writer.add(input)
    }

    func append(_ buffer: CVPixelBuffer, at time: CMTime) {
        if !started {
            guard writer.startWriting() else {
                return
            }
            writer.startSession(atSourceTime: time)
            started = true
        }
        if input.isReadyForMoreMediaData {
            adaptor.append(buffer, withPresentationTime: time)
        }
    }

    func finish(completion: @escaping (URL?) -> Void) {
        guard started else {
            completion(nil)
            return
        }
        input.markAsFinished()
        writer.finishWriting { [writer, url] in
            completion(writer.status == .completed ? url : nil)
        }
    }
}

// One CSV per launch in Documents (visible in the Files app and over USB).
final class FrameLog {
    static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    private let handle: FileHandle?

    init() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let url = Self.documents.appendingPathComponent("frames-\(formatter.string(from: Date())).csv")
        FileManager.default.createFile(
            atPath: url.path,
            contents: Data("pts_ms,interval_ms,vision_ms,render_ms,faces,held,lost,recording\n".utf8))
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    func write(presentation: CMTime, stats: RoundPipeline.Stats, recording: Bool) {
        let line = String(
            format: "%.1f,%.1f,%.2f,%.2f,%d,%d,%d,%d\n",
            presentation.seconds * 1000,
            stats.intervalMs,
            stats.visionMs,
            stats.renderMs,
            stats.faces,
            stats.held ? 1 : 0,
            stats.lost ? 1 : 0,
            recording ? 1 : 0)
        try? handle?.write(contentsOf: Data(line.utf8))
    }
}
