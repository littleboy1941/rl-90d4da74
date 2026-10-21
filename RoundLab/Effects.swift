import CoreImage

// Effect contract shared with the future Telegram port: Telegram's round
// video filter (CameraRoundLegacyVideoFilter) also works on CIImage.
// Faces are pixel rects in the source image (Core Image coordinates,
// origin at the bottom left).
protocol RoundEffect {
    var name: String { get }
    func apply(to image: CIImage, faces: [CGRect], trackingLost: Bool) -> CIImage
}

// Step 1: opaque black oval over every face; whole frame black when the
// face is lost (fail closed).
struct BlackOvalEffect: RoundEffect {
    let name = "black_oval"
    var expansion: CGFloat = 0.25

    func apply(to image: CIImage, faces: [CGRect], trackingLost: Bool) -> CIImage {
        let extent = image.extent
        if trackingLost {
            return CIImage(color: .black).cropped(to: extent)
        }
        var result = image
        for face in faces {
            let rect = face.insetBy(
                dx: -face.width * expansion,
                dy: -face.height * expansion)
            result = Self.ellipse(in: rect, clippedTo: extent).composited(over: result)
        }
        return result
    }

    static func ellipse(in rect: CGRect, clippedTo extent: CGRect) -> CIImage {
        let gradient = CIFilter(name: "CIRadialGradient", parameters: [
            "inputCenter": CIVector(x: 0, y: 0),
            "inputRadius0": 0.98,
            "inputRadius1": 1.0,
            "inputColor0": CIColor(red: 0, green: 0, blue: 0, alpha: 1),
            "inputColor1": CIColor(red: 0, green: 0, blue: 0, alpha: 0),
        ])?.outputImage ?? CIImage(color: .black)
        return gradient
            .transformed(by: CGAffineTransform(scaleX: rect.width / 2, y: rect.height / 2))
            .transformed(by: CGAffineTransform(translationX: rect.midX, y: rect.midY))
            .cropped(to: rect.intersection(extent))
    }
}
