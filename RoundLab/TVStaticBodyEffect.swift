import CoreImage
import Vision

// "TV static" over the whole person (author 2026-10-23: body, arms, legs;
// the room stays as it is): Apple person segmentation finds everyone in
// the frame, the silhouette is filled with the same snow as `tv_static`
// (TVStaticGPU), blown out at the top of the head and thinning into
// separate blobs towards the bottom of the frame, with a glow over the
// edge. Needs no face, so it works on both cameras and from behind.
// Segmentation failure covers the whole frame. PC preview:
// `tools/sim_tvstatic.py --bg camera`.
// Three variants to compare on the phone (author 2026-10-23: "follow the
// user faster, more precise"): `fast` - mask of this very frame (no Vision
// smoothing, which lags behind motion), own hole filler; `accurate` - the
// same with Vision's best quality (build 39 smoothed with Vision instead).
// Build 41 (author on 40: the hair edge and a fast-waving hand show for a
// moment): `motion` also covers whatever moved since the last frame close
// to the person (a blurred hand the segmentation lost), and the mask is
// grown more around each head (`headGrow`, hair edge). Build 40 stays as
// `build40` to compare.
struct TVStaticBodyEffect: RoundEffect {
    let name: String
    let mask: PersonMask
    var motion = true
    var headGrow: CGFloat = 0.07      // extra dilation around a head, of the face width
    var grow: CGFloat = 0.01          // mask dilation, of the frame width
    var glow: Double = 0.45           // glow strength
    var glowRadius: CGFloat = 0.02    // glow sigma, of the frame width

    static let fast = TVStaticBodyEffect(name: "tv_static_body", mask: PersonMask(quality: .balanced, visionSmoothing: false))
    static let accurate = TVStaticBodyEffect(name: "tv_static_body_hq", mask: PersonMask(quality: .accurate, visionSmoothing: false))
    static let build40 = TVStaticBodyEffect(
        name: "tv_static_body_40", mask: PersonMask(quality: .balanced, visionSmoothing: false), motion: false, headGrow: 0)

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        guard var person = self.mask.mask(image) else {
            return image.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
        if motion, let moving = self.mask.motion(image, near: person) {
            person = moving.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: person])
        }
        for face in faces where headGrow > 0 {
            let head = face.rect
                .insetBy(dx: -face.rect.width * 0.45, dy: -face.rect.height * 0.35)
                .offsetBy(dx: 0, dy: face.rect.height * 0.2)
            let wider = person.clampedToExtent()
                .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: max(2, face.rect.width * headGrow)])
                .cropped(to: head.intersection(extent))
            person = wider.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: person])
        }
        let mask = person.cropped(to: extent)
        // Exposure runs from the top of the head (above the highest face,
        // or the top of the frame) down to the bottom of the frame.
        let top = faces.map { $0.rect.maxY + $0.rect.height * 0.35 }.max().map { min($0, extent.maxY) } ?? extent.maxY
        let span = CGRect(x: extent.minX, y: extent.minY, width: extent.width, height: top - extent.minY)
        guard let snow = TVStaticGPU.shared.run(size: extent.size, face: span, time: time)?
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY)) else {
            return image.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
        let soft = mask.clampedToExtent()
            .applyingFilter("CIMorphologyMaximum", parameters: [
                kCIInputRadiusKey: max(2, extent.width * grow),
            ])
            .applyingGaussianBlur(sigma: 1.0)
            .cropped(to: extent)
        let black = CIImage(color: .black).cropped(to: extent)
        let masked = snow.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: black,
            kCIInputMaskImageKey: soft,
        ])
        let halo = masked.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(max(extent.width * glowRadius, 2)))
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: CGFloat(glow) * 0.9, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: CGFloat(glow) * 0.95, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(glow), w: 0),
            ])
            .cropped(to: extent)
        return snow.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: image,
            kCIInputMaskImageKey: soft,
        ])
        .applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: halo])
        .cropped(to: extent)
    }
}

// Person segmentation (Vision) as a grey mask the size of the frame:
// white = person. With `visionSmoothing` the stateful request runs
// through one sequence handler and Vision smooths the mask between frames
// (steady, but trails behind motion). Without it every frame is segmented
// on its own and a pixel seen as person stays covered while it fades out
// over a few frames (`fade` per frame), so one-frame holes never show the
// person and the mask still follows motion at once.
// Called from the video queue only.
final class PersonMask {
    let visionSmoothing: Bool
    var fade = 0.55

    private let handler = VNSequenceRequestHandler()
    private let request = VNGeneratePersonSegmentationRequest()
    private var history: [UInt8] = []
    private var buffers: [CVPixelBuffer] = []
    private var next = 0
    // Motion: the frame shrunk to `motionWidth` px, this and the last one.
    var motionWidth = 120
    var motionThreshold = 0.09        // mean RGB change that counts as motion
    var motionReach: CGFloat = 0.06   // how far from the person, of the frame width
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var frames: [CVPixelBuffer] = []
    private var frameIndex = 0
    private var hasPrevious = false

    init(quality: VNGeneratePersonSegmentationRequest.QualityLevel, visionSmoothing: Bool) {
        self.visionSmoothing = visionSmoothing
        request.qualityLevel = quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    }

    func mask(_ image: CIImage) -> CIImage? {
        let extent = image.extent
        if visionSmoothing {
            guard (try? handler.perform([request], on: image, orientation: .up)) != nil else {
                return nil
            }
        } else {
            guard (try? VNImageRequestHandler(ciImage: image, orientation: .up).perform([request])) != nil else {
                return nil
            }
        }
        guard let segmented = request.results?.first?.pixelBuffer,
              let pixels = visionSmoothing ? segmented : hold(segmented) else {
            return nil
        }
        let mask = CIImage(cvPixelBuffer: pixels)
        return mask
            .transformed(by: CGAffineTransform(
                scaleX: extent.width / mask.extent.width,
                y: extent.height / mask.extent.height))
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
    }

    // White where the frame changed since the last call, only within
    // `motionReach` of `person` (a moving room far away stays as it is);
    // nil on the first frame. Computed on the shrunk frame.
    func motion(_ image: CIImage, near person: CIImage) -> CIImage? {
        let extent = image.extent
        let scale = CGFloat(motionWidth) / extent.width
        let width = motionWidth
        let height = Int((extent.height * scale).rounded())
        if frames.count != 2 || CVPixelBufferGetHeight(frames[0]) != height {
            frames = (0..<2).compactMap { _ in
                var buffer: CVPixelBuffer?
                CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                    [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary, &buffer)
                return buffer
            }
            hasPrevious = false
        }
        guard frames.count == 2 else {
            return nil
        }
        let current = frames[frameIndex]
        let previous = frames[frameIndex ^ 1]
        frameIndex ^= 1
        let toSmall = CGAffineTransform(scaleX: scale, y: scale)
            .translatedBy(x: -extent.minX, y: -extent.minY)
        let small = CGRect(x: 0, y: 0, width: width, height: height)
        context.render(
            image.clampedToExtent()
                .applyingGaussianBlur(sigma: Double(0.5 / scale))
                .cropped(to: extent)
                .transformed(by: toSmall),
            to: current, bounds: small, colorSpace: nil)
        defer { hasPrevious = true }
        guard hasPrevious else {
            return nil
        }
        let third = CGFloat(1) / 3
        let moving = CIImage(cvPixelBuffer: current)
            .applyingFilter("CIDifferenceBlendMode", parameters: [
                kCIInputBackgroundImageKey: CIImage(cvPixelBuffer: previous),
            ])
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: third, y: third, z: third, w: 0),
                "inputGVector": CIVector(x: third, y: third, z: third, w: 0),
                "inputBVector": CIVector(x: third, y: third, z: third, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            ])
            .applyingFilter("CIColorThreshold", parameters: ["inputThreshold": motionThreshold])
            .clampedToExtent()
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 2])
            .cropped(to: small)
        let near = person
            .transformed(by: toSmall)
            .clampedToExtent()
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: CGFloat(width) * motionReach])
            .cropped(to: small)
        return moving
            .applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: near])
            .transformed(by: toSmall.inverted())
            .cropped(to: extent)
    }

    // max(this frame, last result faded), into one of two own buffers.
    private func hold(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        if history.count != width * height {
            history = [UInt8](repeating: 0, count: width * height)
            buffers = (0..<2).compactMap { _ in
                var buffer: CVPixelBuffer?
                CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent8,
                                    [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary, &buffer)
                return buffer
            }
        }
        guard buffers.count == 2 else {
            return nil
        }
        let target = buffers[next]
        next ^= 1
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(target, [])
        defer {
            CVPixelBufferUnlockBaseAddress(target, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let input = CVPixelBufferGetBaseAddress(source)?.assumingMemoryBound(to: UInt8.self),
              let output = CVPixelBufferGetBaseAddress(target)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        let inRow = CVPixelBufferGetBytesPerRow(source)
        let outRow = CVPixelBufferGetBytesPerRow(target)
        let keep = UInt16(fade * 256)
        history.withUnsafeMutableBufferPointer { last in
            for y in 0..<height {
                for x in 0..<width {
                    let i = y * width + x
                    let value = max(input[y * inRow + x], UInt8((UInt16(last[i]) * keep) >> 8))
                    last[i] = value
                    output[y * outRow + x] = value
                }
            }
        }
        return target
    }
}
