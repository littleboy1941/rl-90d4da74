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
struct TVStaticBodyEffect: RoundEffect {
    let name = "tv_static_body"
    var grow: CGFloat = 0.015         // mask dilation, of the frame width
    var glow: Double = 0.45           // glow strength
    var glowRadius: CGFloat = 0.02    // glow sigma, of the frame width

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        guard let mask = PersonMask.shared.mask(image) else {
            return image.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
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

// Person segmentation (Vision, balanced quality) as a grey mask the size
// of the frame: white = person. The request is stateful and runs through
// one sequence handler, so Vision smooths the mask between frames.
// Called from the video queue only.
final class PersonMask {
    static let shared = PersonMask()

    private let handler = VNSequenceRequestHandler()
    private let request: VNGeneratePersonSegmentationRequest = {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }()

    func mask(_ image: CIImage) -> CIImage? {
        let extent = image.extent
        guard (try? handler.perform([request], on: image, orientation: .up)) != nil,
              let pixels = request.results?.first?.pixelBuffer else {
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
}
