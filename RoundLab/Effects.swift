import CoreGraphics
import CoreImage

// One tracked face in source pixel coordinates (Core Image coordinates,
// origin at the bottom left). Contour runs along the jaw from ear to ear;
// empty when Vision returned no landmarks.
struct Face {
    var rect: CGRect
    var contour: [CGPoint]
}

// Effect contract shared with the future Telegram port: Telegram's round
// video filter (CameraRoundLegacyVideoFilter) also works on CIImage.
protocol RoundEffect {
    var name: String { get }
    func apply(to image: CIImage, faces: [Face], trackingLost: Bool) -> CIImage
}

// Step 1: opaque black oval over every face; whole frame black when the
// face is lost (fail closed).
struct BlackOvalEffect: RoundEffect {
    let name = "black_oval"
    var expansion: CGFloat = 0.25

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool) -> CIImage {
        let extent = image.extent
        if trackingLost {
            return CIImage(color: .black).cropped(to: extent)
        }
        let mask = Mask(extent: extent)
        for face in faces {
            let rect = face.rect.insetBy(
                dx: -face.rect.width * expansion,
                dy: -face.rect.height * expansion)
            mask.context?.fillEllipse(in: rect)
        }
        return mask.image(over: image)
    }
}

// Step 2: opaque black silhouette along the face contour (jaw from Vision
// landmarks) closed over the forehead with an arc, plus a margin.
struct BlackSilhouetteEffect: RoundEffect {
    let name = "black_silhouette"
    var margin: CGFloat = 0.12      // of the face width, all around
    var foreheadLift: CGFloat = 0.35 // of the face height, above the box

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool) -> CIImage {
        let extent = image.extent
        if trackingLost {
            return CIImage(color: .black).cropped(to: extent)
        }
        let mask = Mask(extent: extent)
        guard let context = mask.context else {
            return CIImage(color: .black).cropped(to: extent)
        }
        for face in faces {
            let path = Self.silhouette(face, lift: foreheadLift)
            context.addPath(path)
            context.fillPath()
            context.setLineJoin(.round)
            context.setLineWidth(face.rect.width * margin * 2)
            context.addPath(path)
            context.strokePath()
        }
        return mask.image(over: image)
    }

    static func silhouette(_ face: Face, lift: CGFloat) -> CGPath {
        let path = CGMutablePath()
        let rect = face.rect
        guard face.contour.count >= 3,
              let first = face.contour.first,
              let last = face.contour.last else {
            path.addEllipse(in: rect.insetBy(dx: 0, dy: -rect.height * lift / 2))
            return path
        }
        path.move(to: first)
        for point in face.contour.dropFirst() {
            path.addLine(to: point)
        }
        // Forehead: arc from the last contour point over the top back to the first.
        let top = CGPoint(x: (first.x + last.x) / 2, y: rect.maxY + rect.height * lift)
        path.addQuadCurve(
            to: CGPoint(x: (last.x + top.x) / 2, y: top.y),
            control: CGPoint(x: last.x, y: top.y))
        path.addQuadCurve(
            to: first,
            control: CGPoint(x: first.x, y: top.y))
        path.closeSubpath()
        return path
    }
}

// Black-on-transparent mask drawn with Core Graphics (same coordinates as
// Core Image: origin at the bottom left), composited over the frame.
final class Mask {
    let extent: CGRect
    let context: CGContext?

    init(extent: CGRect) {
        self.extent = extent
        context = CGContext(
            data: nil,
            width: Int(extent.width),
            height: Int(extent.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.translateBy(x: -extent.minX, y: -extent.minY)
        context?.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context?.setStrokeColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    }

    func image(over source: CIImage) -> CIImage {
        guard let cgImage = context?.makeImage() else {
            return CIImage(color: .black).cropped(to: extent)
        }
        let overlay = CIImage(cgImage: cgImage)
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
        return overlay.composited(over: source)
    }
}

let allEffects: [RoundEffect] = [BlackSilhouetteEffect(), BlackOvalEffect()]
