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

// "Leak" (reference 2, Elliot poster): the real face stays visible; one
// side of it leaks sideways into the dark in ragged horizontal streaks —
// each band of rows takes the color at the face edge and stretches it out
// for a random, flickering length. Low-key grade, CRT lines, color fringe
// and a terminal prompt on top.
struct LeakStyle: FaceStyle {
    let name = "leak"
    var bandHeight: CGFloat = 3
    var maxLength: CGFloat = 0.9      // of the face width
    var emptyShare = 0.3              // bands without a streak
    var ticksPerSecond = 14.0         // how often streak lengths change

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
        // Low key: everything but the face falls into the dark.
        let soft = overlay.clampedToExtent().applyingGaussianBlur(sigma: 14).cropped(to: extent)
        let graded = camera.applyingFilter("CIColorControls", parameters: [
            kCIInputSaturationKey: 0.55,
            kCIInputContrastKey: 1.15,
        ])
        let dark = graded.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -2.2])
        var image = graded.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: dark,
            kCIInputMaskImageKey: soft,
        ]).cropped(to: extent)

        let bounds = face.bounds
        if !bounds.isNull, bounds.width > 4 {
            image = Self.streaks(over: image, face: face, bounds: bounds, style: self, time: time)
        }
        return finish(image, time: time)
    }

    static func streaks(over image: CIImage, face: FaceShape, bounds: CGRect, style: LeakStyle, time: Double) -> CIImage {
        let extent = image.extent
        let rows = Int(extent.height / style.bandHeight) + 1
        var edges = [CGFloat](repeating: .greatestFiniteMagnitude, count: rows)
        for point in face.points {
            let row = Int(point.y / style.bandHeight)
            if row >= 0, row < rows {
                edges[row] = min(edges[row], point.x)
            }
        }
        let tick = UInt64(max(0, Int(time * style.ticksPerSecond)))
        var streaks = CIImage.empty()
        for row in 0..<rows where edges[row] < .greatestFiniteMagnitude {
            var generator = SplitMix(seed: UInt64(row) &* 7919 &+ tick &* 104_729)
            let roll = Double(generator.next() % 10_000) / 10_000
            guard roll >= style.emptyShare else {
                continue
            }
            let strength = (roll - style.emptyShare) / (1 - style.emptyShare)
            let length = bounds.width * style.maxLength * CGFloat(strength * strength)
            guard length >= 2 else {
                continue
            }
            let edge = edges[row]
            let sampleX = floor(min(edge + bounds.width * 0.05, extent.maxX - 1))
            let y = CGFloat(row) * style.bandHeight
            let column = image
                .cropped(to: CGRect(x: sampleX, y: y, width: 1, height: style.bandHeight))
                .samplingNearest()
                .transformed(by: CGAffineTransform(translationX: -sampleX, y: 0))
                .transformed(by: CGAffineTransform(scaleX: length, y: 1))
                .transformed(by: CGAffineTransform(translationX: edge - length, y: 0))
            streaks = column.composited(over: streaks)
        }
        return streaks.composited(over: image).cropped(to: extent)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        let extent = image.extent
        // Cold tint, color fringe, CRT lines, vignette, terminal prompt.
        var result = image.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.92, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0.97, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 1.06, w: 0),
        ])
        result = SignalEffect.colorSplit(result, extent: extent, shift: 1.5)
        result = SignalEffect.scanlines(extent: extent).composited(over: result)
        result = result.applyingFilter("CIVignette", parameters: [
            kCIInputIntensityKey: 0.9,
            kCIInputRadiusKey: 1.4,
        ])
        if let prompt = Self.prompt(extent: extent, time: time) {
            result = prompt.composited(over: result)
        }
        return result.cropped(to: extent)
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
