import ARKit
import CoreImage
import SceneKit
import UIKit

// "Face of text" (reference 1): the face mesh is covered with an opaque
// texture of terminal lines on black, lit by a key light so the head shape
// reads; the text slowly scrolls. Eyes and mouth are closed (fill mesh),
// so nothing of the real face shows through.
struct TextFaceStyle: FaceStyle {
    let name = "text_face"
    var tiles: Float = 3
    var scrollPerSecond: Double = 0.03

    static let texture: UIImage = makeTexture()

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        guard let mesh = ARSCNFaceGeometry(device: device, fillMesh: true) else {
            return []
        }
        let material = SCNMaterial()
        material.diffuse.contents = Self.texture
        material.diffuse.wrapS = .repeat
        material.diffuse.wrapT = .repeat
        material.diffuse.contentsTransform = SCNMatrix4MakeScale(tiles, tiles, 1)
        material.lightingModel = .lambert
        material.isDoubleSided = false
        mesh.firstMaterial = material
        let node = SCNNode(geometry: mesh)
        // A little larger than the face so the edges stay covered.
        return [StyledNode(node: node, scale: 1.06)]
    }

    func update(_ nodes: [StyledNode], time: Double) {
        let offset = Float((time * scrollPerSecond).truncatingRemainder(dividingBy: 1))
        let transform = SCNMatrix4Mult(
            SCNMatrix4MakeScale(tiles, tiles, 1),
            SCNMatrix4MakeTranslation(0, offset, 0))
        for styled in nodes {
            styled.node.geometry?.firstMaterial?.diffuse.contentsTransform = transform
        }
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        let extent = image.extent
        return SignalEffect.scanlines(extent: extent).composited(over: image).cropped(to: extent)
    }

    // Monospaced terminal lines on black; a few words in signal red.
    static func makeTexture() -> UIImage {
        let side: CGFloat = 1024
        let phrases = [
            "hello friend", "root@elliot:~#", "control is an illusion", "fsociety",
            "0x7f3a 0x00ff 0xdead 0xbeef", "ssh -p 2222", "we are finally awake",
            "kill -9", "./rsi --recursive", "01001000 01101001", "chmod 000",
            "are you seeing this too", "SIGNAL", "exec", "nmap -sV", "rm -rf /",
        ]
        let font = UIFont(name: "Menlo-Bold", size: 17) ?? .monospacedSystemFont(ofSize: 17, weight: .bold)
        let white = UIColor(white: 0.9, alpha: 1)
        let red = UIColor(red: 0.95, green: 0.12, blue: 0.15, alpha: 1)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side))
        return renderer.image { context in
            UIColor.black.setFill()
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
            var generator = SplitMix(seed: 2026)
            var y: CGFloat = 2
            while y < side {
                var x: CGFloat = CGFloat(generator.next() % 40) - 20
                while x < side {
                    let phrase = phrases[Int(generator.next() % UInt64(phrases.count))]
                    let color = generator.next() % 9 == 0 ? red : white
                    let text = NSAttributedString(string: phrase + "  ", attributes: [
                        .font: font,
                        .foregroundColor: color,
                    ])
                    text.draw(at: CGPoint(x: x, y: y))
                    x += text.size().width
                }
                y += font.lineHeight * 0.92
            }
        }
    }
}

// "Leak" (reference 2, Elliot poster): the real face stays visible, almost
// blown out to white on black; its left side leaks into the dark in dense
// horizontal streaks one LED row tall, some rows inside the face slide
// sideways, and the whole picture is shown as a low-res LED dot matrix.
struct LeakStyle: FaceStyle {
    let name = "leak"
    var cell: CGFloat = 5             // LED dot pitch in output pixels
    var maxLength: CGFloat = 1.15     // of the face width
    var emptyShare = 0.12             // rows without a streak
    var shiftedShare = 0.08           // rows inside the face slid sideways
    var ticksPerSecond = 12.0         // how often the streaks change

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        // White unlit face: used only as a soft mask for the lighting.
        guard let mesh = ARSCNFaceGeometry(device: device, fillMesh: true) else {
            return []
        }
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.white
        material.lightingModel = .constant
        mesh.firstMaterial = material
        return [StyledNode(node: SCNNode(geometry: mesh), scale: 1.0)]
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        // Hard low key: the face close to white, the rest close to black.
        let soft = overlay.clampedToExtent().applyingGaussianBlur(sigma: 10).cropped(to: extent)
        let graded = camera.applyingFilter("CIColorControls", parameters: [
            kCIInputSaturationKey: 0.45,
            kCIInputContrastKey: 1.5,
            kCIInputBrightnessKey: 0.03,
        ])
        let bright = graded.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: 0.45])
        let dark = graded.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -2.8])
        var image = bright.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: dark,
            kCIInputMaskImageKey: soft,
        ]).cropped(to: extent)

        let bounds = face.bounds
        if !bounds.isNull, bounds.width > 4 {
            image = streaks(over: image, face: face, bounds: bounds, time: time)
        }
        return finish(image, time: time)
    }

    func streaks(over image: CIImage, face: FaceShape, bounds: CGRect, time: Double) -> CIImage {
        let extent = image.extent
        let rows = Int(extent.height / cell) + 1
        var left = [CGFloat](repeating: .greatestFiniteMagnitude, count: rows)
        var right = [CGFloat](repeating: -.greatestFiniteMagnitude, count: rows)
        for point in face.points {
            let row = Int(point.y / cell)
            if row >= 0, row < rows {
                left[row] = min(left[row], point.x)
                right[row] = max(right[row], point.x)
            }
        }
        let tick = UInt64(max(0, Int(time * ticksPerSecond)))
        var layer = CIImage.empty()
        for row in 0..<rows where left[row] < .greatestFiniteMagnitude {
            var generator = SplitMix(seed: UInt64(row) &* 7919 &+ tick &* 104_729)
            let roll = Double(generator.next() % 10_000) / 10_000
            let y = CGFloat(row) * cell
            let band = CGRect(x: extent.minX, y: y, width: extent.width, height: cell)

            // Some rows inside the face slide sideways (glitch).
            if roll < shiftedShare {
                let shift = -cell * CGFloat(2 + generator.next() % 6)
                let span = CGRect(x: left[row], y: y, width: max(right[row] - left[row], 1), height: cell)
                let moved = image.cropped(to: span)
                    .transformed(by: CGAffineTransform(translationX: shift, y: 0))
                    .cropped(to: band)
                layer = moved.composited(over: layer)
            }
            guard roll >= emptyShare else {
                continue
            }
            // Streak: the color just inside the left edge, stretched out to the left.
            let strength = (roll - emptyShare) / (1 - emptyShare)
            let length = bounds.width * maxLength * CGFloat(0.2 + 0.8 * strength)
            let edge = left[row]
            let inside = bounds.width * (0.04 + 0.1 * Double(generator.next() % 100) / 100)
            let sampleX = floor(min(edge + inside, extent.maxX - 1))
            let streak = image
                .cropped(to: CGRect(x: sampleX, y: y, width: 1, height: cell))
                .samplingNearest()
                .transformed(by: CGAffineTransform(translationX: -sampleX, y: 0))
                .transformed(by: CGAffineTransform(scaleX: length + inside, y: 1))
                .transformed(by: CGAffineTransform(translationX: edge - length, y: 0))
            // Tips go teal and fade, like the reference.
            let tip = CGRect(x: edge - length, y: y, width: length * 0.35, height: cell)
            let teal = streak.cropped(to: tip).applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0.35, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 0.8, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0.75, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.8),
            ])
            layer = teal.composited(over: streak.composited(over: layer))
        }
        return layer.composited(over: image).cropped(to: extent)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        let extent = image.extent
        // Faint red fringe, then the LED dot matrix, then the prompt on top.
        var result = SignalEffect.colorSplit(image, extent: extent, shift: 2)
        result = Self.ledMatrix(result, extent: extent, cell: cell)
        result = result.applyingFilter("CIVignette", parameters: [
            kCIInputIntensityKey: 0.7,
            kCIInputRadiusKey: 1.5,
        ])
        if let prompt = Self.prompt(extent: extent, time: time) {
            result = prompt.composited(over: result)
        }
        return result.cropped(to: extent)
    }

    // Low-res picture shown as square LED dots with dark gaps.
    static func ledMatrix(_ image: CIImage, extent: CGRect, cell: CGFloat) -> CIImage {
        let blocks = image.applyingFilter("CIPixellate", parameters: [
            kCIInputCenterKey: CIVector(x: cell / 2, y: cell / 2),
            kCIInputScaleKey: cell,
        ])
        return blocks
            .applyingFilter("CIMultiplyCompositing", parameters: [
                kCIInputBackgroundImageKey: dotTile(cell: Int(cell)),
            ])
            .cropped(to: extent)
    }

    // One LED: a rounded bright square with a dark gap around it, tiled.
    static func dotTile(cell: Int) -> CIImage {
        let size = CGFloat(cell)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: {
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            format.opaque = true
            return format
        }())
        let image = renderer.image { context in
            UIColor(white: 0.12, alpha: 1).setFill()
            context.fill(CGRect(x: 0, y: 0, width: size, height: size))
            UIColor.white.setFill()
            UIBezierPath(
                roundedRect: CGRect(x: 0.5, y: 0.5, width: size - 1.5, height: size - 1.5),
                cornerRadius: size * 0.2).fill()
        }
        guard let cgImage = image.cgImage else {
            return CIImage(color: .white)
        }
        return CIImage(cgImage: cgImage).applyingFilter("CIAffineTile", parameters: [
            kCIInputTransformKey: NSValue(cgAffineTransform: .identity),
        ])
    }

    // "root@elliot:~# _" near the top of the circle, cursor blinking.
    static func prompt(extent: CGRect, time: Double) -> CIImage? {
        let cursor = Int(time * 2) % 2 == 0 ? "_" : " "
        guard let text = CIFilter(name: "CITextImageGenerator", parameters: [
            "inputText": "root@elliot:~# " + cursor,
            "inputFontName": "Menlo",
            "inputFontSize": extent.width * 0.04,
            "inputScaleFactor": 1,
        ])?.outputImage else {
            return nil
        }
        let tinted = text.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBiasVector": CIVector(x: 0.85, y: 0.88, z: 0.9, w: 0),
        ])
        return tinted.transformed(by: CGAffineTransform(
            translationX: extent.midX - tinted.extent.width / 2 - tinted.extent.minX,
            y: extent.minY + extent.height * 0.8 - tinted.extent.minY))
    }
}

// Small deterministic generator so the texture is the same every launch.
struct SplitMix {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
