import CoreGraphics
import CoreImage

// One tracked face in source pixel coordinates (Core Image coordinates,
// origin at the bottom left). Contour runs along the jaw from ear to ear;
// empty when Vision returned no landmarks. `hold` is 0 for a face seen in
// this frame and grows to 1 while the last seen face is held after a miss.
struct Face {
    var rect: CGRect
    var contour: [CGPoint]
    var hold: CGFloat = 0
}

// Effect contract shared with the future Telegram port: Telegram's round
// video filter (CameraRoundLegacyVideoFilter) also works on CIImage.
// `time` is in seconds and drives animation.
protocol RoundEffect {
    var name: String { get }
    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage
}

// Step 1: opaque black oval over every face; whole frame black when the
// face is lost (fail closed).
struct BlackOvalEffect: RoundEffect {
    let name = "black_oval"
    var expansion: CGFloat = 0.25

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
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

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        if trackingLost {
            return CIImage(color: .black).cropped(to: extent)
        }
        return Silhouette.mask(faces, extent: extent).image(over: image)
    }
}

// Step 3: "signal" in the Mr. Robot spirit. The silhouette is filled with
// moving synthetic static (no face pixels), the whole frame gets CRT lines,
// a slight color split and a vignette; now and then a horizontal band of
// the already covered frame slides sideways. Lost face: static + SIGNAL LOST.
struct SignalEffect: RoundEffect {
    let name = "signal"

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        let noise = Self.staticNoise(extent: extent, time: time, brightness: 0.22)
        var frame: CIImage
        if trackingLost {
            frame = Self.lostScreen(extent: extent, noise: noise, time: time)
        } else {
            let mask = Silhouette.mask(faces, extent: extent)
            guard let alpha = mask.ciImage() else {
                return CIImage(color: .black).cropped(to: extent)
            }
            // Black first (guarantee), static on top, both only inside the mask.
            let covered = noise.composited(over: CIImage(color: .black).cropped(to: extent))
            frame = covered.applyingFilter("CIBlendWithAlphaMask", parameters: [
                kCIInputBackgroundImageKey: image,
                kCIInputMaskImageKey: alpha,
            ]).cropped(to: extent)
        }
        frame = Self.glitchBand(frame, extent: extent, time: time)
        frame = Self.colorSplit(frame, extent: extent, shift: 2)
        frame = Self.scanlines(extent: extent).composited(over: frame)
        frame = frame.applyingFilter("CIVignette", parameters: [
            kCIInputIntensityKey: 0.8,
            kCIInputRadiusKey: 1.6,
        ])
        return frame.cropped(to: extent)
    }

    // Grey static stretched into horizontal streaks, moving every frame.
    static func staticNoise(extent: CGRect, time: Double, brightness: CGFloat) -> CIImage {
        let random = CIFilter(name: "CIRandomGenerator")?.outputImage ?? CIImage(color: .black)
        let offset = CGFloat(Int(time * 30) % 997) * 37
        let grey = random
            .transformed(by: CGAffineTransform(translationX: offset, y: offset * 0.61))
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: brightness, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: brightness, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: brightness, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            ])
            .transformed(by: CGAffineTransform(scaleX: 6, y: 1))
        return grey.cropped(to: extent)
    }

    static func lostScreen(extent: CGRect, noise: CIImage, time: Double) -> CIImage {
        let base = noise.composited(over: CIImage(color: .black).cropped(to: extent))
        let blink = Int(time * 2) % 2 == 0
        guard blink,
              let text = CIFilter(name: "CITextImageGenerator", parameters: [
                  "inputText": "SIGNAL LOST",
                  "inputFontName": "Menlo-Bold",
                  "inputFontSize": extent.width * 0.075,
                  "inputScaleFactor": 1,
              ])?.outputImage else {
            return base
        }
        let white = text.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBiasVector": CIVector(x: 0.92, y: 0.92, z: 0.9, w: 0),
        ])
        let placed = white.transformed(by: CGAffineTransform(
            translationX: extent.midX - white.extent.width / 2 - white.extent.minX,
            y: extent.midY - white.extent.height / 2 - white.extent.minY))
        return placed.composited(over: base)
    }

    // For ~150 ms every ~1.7 s one horizontal band slides sideways.
    static func glitchBand(_ image: CIImage, extent: CGRect, time: Double) -> CIImage {
        let period = 1.7
        let phase = time.truncatingRemainder(dividingBy: period)
        guard phase < 0.15 else {
            return image
        }
        let seed = Int(time / period)
        let bandHeight = extent.height * CGFloat(0.06 + Double(seed % 5) * 0.03)
        let bandY = extent.minY + extent.height * CGFloat((seed * 37) % 80) / 100
        let shift = extent.width * CGFloat((seed % 2 == 0 ? 1 : -1) * (6 + seed % 9)) / 100
        let band = CGRect(x: extent.minX, y: bandY, width: extent.width, height: bandHeight)
        let moved = image
            .transformed(by: CGAffineTransform(translationX: shift, y: 0))
            .cropped(to: band)
        let gap = CIImage(color: .black).cropped(to: band)
        return moved.composited(over: gap.composited(over: image))
    }

    // Red shifted right, blue shifted left, green in place.
    static func colorSplit(_ image: CIImage, extent: CGRect, shift: CGFloat) -> CIImage {
        func channel(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, dx: CGFloat) -> CIImage {
            image.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: r, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: g, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: b, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            ]).transformed(by: CGAffineTransform(translationX: dx, y: 0))
        }
        let red = channel(1, 0, 0, dx: shift)
        let green = channel(0, 1, 0, dx: 0)
        let blue = channel(0, 0, 1, dx: -shift)
        return red
            .applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: green])
            .applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: blue])
            .cropped(to: extent)
    }

    // Thin dark horizontal lines every 3 px.
    static func scanlines(extent: CGRect) -> CIImage {
        let stripes = CIFilter(name: "CIStripesGenerator", parameters: [
            "inputColor0": CIColor(red: 0, green: 0, blue: 0, alpha: 0.28),
            "inputColor1": CIColor(red: 0, green: 0, blue: 0, alpha: 0),
            "inputWidth": 1.5,
            "inputSharpness": 1,
        ])?.outputImage ?? CIImage(color: .clear)
        return stripes
            .transformed(by: CGAffineTransform(rotationAngle: .pi / 2))
            .cropped(to: extent)
    }
}

// Face silhouettes: jaw contour + forehead arc, filled and stroked with a margin.
enum Silhouette {
    static let margin: CGFloat = 0.12       // of the face width, all around
    static let foreheadLift: CGFloat = 0.35 // of the face height, above the box

    static func mask(_ faces: [Face], extent: CGRect) -> Mask {
        let mask = Mask(extent: extent)
        guard let context = mask.context else {
            return mask
        }
        for face in faces {
            let path = Self.path(face)
            context.addPath(path)
            context.fillPath()
            context.setLineJoin(.round)
            context.setLineWidth(face.rect.width * margin * 2)
            context.addPath(path)
            context.strokePath()
        }
        return mask
    }

    static func path(_ face: Face) -> CGPath {
        let path = CGMutablePath()
        let rect = face.rect
        guard face.contour.count >= 3,
              let first = face.contour.first,
              let last = face.contour.last else {
            path.addEllipse(in: rect.insetBy(dx: 0, dy: -rect.height * foreheadLift / 2))
            return path
        }
        path.move(to: first)
        for point in face.contour.dropFirst() {
            path.addLine(to: point)
        }
        // Forehead: arc from the last contour point over the top back to the first.
        let top = CGPoint(x: (first.x + last.x) / 2, y: rect.maxY + rect.height * foreheadLift)
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
// Any failure turns into a fully black frame (fail closed).
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

    func ciImage() -> CIImage? {
        guard let cgImage = context?.makeImage() else {
            return nil
        }
        return CIImage(cgImage: cgImage)
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
    }

    func image(over source: CIImage) -> CIImage {
        guard let overlay = ciImage() else {
            return CIImage(color: .black).cropped(to: extent)
        }
        return overlay.composited(over: source)
    }
}

// Real censoring blur (author's choice 2026-10-22): every head in the frame
// (face box widened to hair and ears) under a strong soft blur with a
// feathered edge. Irreversible: the frame is shrunk 20x before blurring,
// so a 200 px face keeps ~10 px of color and nothing can be unblurred.
// A held face (missed for < 0.5 s) keeps its place and grows; after that
// the whole frame is blurred while a person (body) is still in view but
// no face is found (fail closed). Nobody in view: the frame is left as is.
struct BlurEffect: RoundEffect {
    let name = "blur"
    var shrink: CGFloat = 20          // frame downscale before blurring
    var headWidth: CGFloat = 1.6      // of the face box width
    var headHeight: CGFloat = 1.9     // of the face box height
    var headLift: CGFloat = 0.12      // oval center above the box center, of its height
    var holdGrowth: CGFloat = 0.5     // extra size at the end of a hold
    var feather: CGFloat = 0.08       // edge softness, of the head width

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        let blurred = Self.heavyBlur(image, shrink: shrink)
        if trackingLost {
            return blurred
        }
        // Feather outward only: each oval is drawn larger by 2 sigma, so after
        // blurring the mask the oval itself is still fully covered.
        let mask = Mask(extent: extent)
        var widest: CGFloat = 0
        for face in faces {
            let grow = 1 + holdGrowth * min(max(face.hold, 0), 1)
            let width = face.rect.width * headWidth * grow
            let height = face.rect.height * headHeight * grow
            let pad = width * feather * 2
            let center = CGPoint(x: face.rect.midX, y: face.rect.midY + face.rect.height * headLift)
            mask.context?.fillEllipse(in: CGRect(
                x: center.x - width / 2 - pad, y: center.y - height / 2 - pad,
                width: width + pad * 2, height: height + pad * 2))
            widest = max(widest, width)
        }
        guard let alpha = mask.ciImage() else {
            return blurred
        }
        let soft = alpha.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(max(widest * feather, 1)))
            .cropped(to: extent)
        return blurred.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: image,
            kCIInputMaskImageKey: soft,
        ]).cropped(to: extent)
    }

    // Shrink (area average), blur a little at the small size, scale back up
    // smoothly, then soften the upscale steps.
    static func heavyBlur(_ image: CIImage, shrink: CGFloat) -> CIImage {
        let extent = image.extent
        let small = image.clampedToExtent()
            .cropped(to: extent)
            .applyingFilter("CILanczosScaleTransform", parameters: [
                kCIInputScaleKey: 1 / shrink,
                kCIInputAspectRatioKey: 1,
            ])
            .clampedToExtent()
            .applyingGaussianBlur(sigma: 1.2)
        return small
            .transformed(by: CGAffineTransform(scaleX: shrink, y: shrink))
            .applyingGaussianBlur(sigma: Double(shrink) * 0.6)
            .cropped(to: extent)
    }
}

let allEffects: [RoundEffect] = [BlurEffect(), SignalEffect(), BlackSilhouetteEffect(), BlackOvalEffect()]
