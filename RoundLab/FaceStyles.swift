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
