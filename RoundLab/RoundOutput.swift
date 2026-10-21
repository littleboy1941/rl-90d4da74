import AVFoundation
import CoreMedia

// Shared sink for every frame source: 400x400 processed frames go to the
// preview (as sample buffers), to the recorder and to the CSV log.
final class RoundOutput {
    struct Stats {
        var intervalMs: Double
        var trackMs: Double
        var renderMs: Double
        var faces: Int
        var held: Bool
        var lost: Bool
    }

    let side = 400
    // Called on the source queue.
    var onSample: ((CMSampleBuffer, Stats) -> Void)?

    private let lock = NSLock()
    private var writer: RoundWriter?
    private let log = FrameLog()
    private var pool: CVPixelBufferPool?

    var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return writer != nil
    }

    func startRecording() {
        lock.lock()
        defer { lock.unlock() }
        if writer == nil {
            writer = try? RoundWriter(side: side)
        }
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        lock.lock()
        let current = writer
        writer = nil
        lock.unlock()
        guard let current else {
            completion(nil)
            return
        }
        current.finish(completion: completion)
    }

    func makeBuffer() -> CVPixelBuffer? {
        lock.lock()
        defer { lock.unlock() }
        if pool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: side,
                kCVPixelBufferHeightKey as String: side,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
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

    func deliver(_ buffer: CVPixelBuffer, at time: CMTime, duration: CMTime, stats: Stats) {
        lock.lock()
        let current = writer
        lock.unlock()
        current?.append(buffer, at: time)
        log.write(presentation: time, stats: stats, recording: current != nil)
        if let sample = Self.makeSample(buffer, at: time, duration: duration) {
            onSample?(sample, stats)
        }
    }

    static func makeSample(_ buffer: CVPixelBuffer, at time: CMTime, duration: CMTime) -> CMSampleBuffer? {
        var format: CMVideoFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: nil,
            imageBuffer: buffer,
            formatDescriptionOut: &format)
        guard let format else {
            return nil
        }
        var timing = CMSampleTimingInfo(
            duration: duration,
            presentationTimeStamp: time,
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
            contents: Data("pts_ms,interval_ms,track_ms,render_ms,faces,held,lost,recording\n".utf8))
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    func write(presentation: CMTime, stats: RoundOutput.Stats, recording: Bool) {
        let line = String(
            format: "%.1f,%.1f,%.2f,%.2f,%d,%d,%d,%d\n",
            presentation.seconds * 1000,
            stats.intervalMs,
            stats.trackMs,
            stats.renderMs,
            stats.faces,
            stats.held ? 1 : 0,
            stats.lost ? 1 : 0,
            recording ? 1 : 0)
        try? handle?.write(contentsOf: Data(line.utf8))
    }
}
