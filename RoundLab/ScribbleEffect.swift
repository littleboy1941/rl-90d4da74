import CoreGraphics
import CoreImage

// "Scratched out" (author's reference `docs/ref_scratch.jpg` in fff, drawn
// in black): every face is crossed out with horizontal marker strokes -
// solid in the middle, thinner strokes with ragged ends at the rim, the
// hair left visible. The strokes "boil" like drawn animation: four
// pre-drawn patches switch 8 times a second, each switch with a small
// shift and turn. Vision faces, so both cameras. PC preview with the same
// drawing: `tools/sim_scribble.py --style marker`.
struct ScribbleEffect: RoundEffect {
    let name = "scribble"
    var boilFPS = 8.0
    var size: CGFloat = 1.9           // patch side, of the Vision box height (box: brows to chin)
    var lift: CGFloat = 0.21          // patch centre above the box centre, of its height
    var holdGrowth: CGFloat = 0.3     // extra size at the end of a hold

    static let patchSide = 512
    static let patches: [CIImage] = (0..<4).compactMap { makePatch(seed: UInt64($0)) }

    func apply(to image: CIImage, faces: [Face], trackingLost: Bool, time: Double) -> CIImage {
        let extent = image.extent
        guard !faces.isEmpty, !Self.patches.isEmpty else {
            return image
        }
        // Same boil frame -> same patch, shift and turn.
        let frame = Int(time * boilFPS)
        var random = SplitMix(seed: UInt64(frame) &* 0x9E37_79B1 &+ 11)
        let patch = Self.patches[frame % Self.patches.count]
        let shiftX = Self.normal(&random) * 0.013
        let shiftY = Self.normal(&random) * 0.013
        let turn = Self.normal(&random) * 3 * .pi / 180
        var mask: CIImage?
        for face in faces {
            let grow = 1 + holdGrowth * min(max(face.hold, 0), 1)
            let side = face.rect.height * size * grow
            let center = CGPoint(x: face.rect.midX + shiftX * side,
                                 y: face.rect.midY + lift * face.rect.height + shiftY * side)
            let half = CGFloat(Self.patchSide) / 2
            let scale = side / CGFloat(Self.patchSide)
            let placed = patch
                .transformed(by: CGAffineTransform(translationX: -half, y: -half))
                .transformed(by: CGAffineTransform(rotationAngle: turn))
                .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                .transformed(by: CGAffineTransform(translationX: center.x, y: center.y))
            mask = mask.map {
                placed.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: $0])
            } ?? placed
        }
        guard let mask else {
            return image
        }
        let ink = CIImage(color: CIColor(red: 0.02, green: 0.02, blue: 0.02)).cropped(to: extent)
        return ink.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: image,
            kCIInputMaskImageKey: mask,
        ]).cropped(to: extent)
    }

    // White strokes on black, `patchSide` square: horizontal strokes over an
    // oval (radius ~0.33 x 0.38 of the side), each with a slight slope and
    // wobble, ends overshooting the oval at random and tapering; dense in
    // the middle, thinner at the rim.
    static func makePatch(seed: UInt64) -> CIImage? {
        let side = patchSide
        guard let context = CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return nil
        }
        let big = CGFloat(side)
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: big, height: big))
        context.setFillColor(gray: 1, alpha: 1)
        var random = SplitMix(seed: seed &* 7919 &+ 3)
        let c = big / 2
        let rx = big * 0.33
        let ry = big * 0.38
        for _ in 0..<160 {
            let v = min(max(normal(&random) * 0.5, -1), 1)
            let y = c + v * ry
            let half = rx * sqrt(max(0, 1 - v * v)) * uniform(&random, 0.85, 1.45)
            let shift = normal(&random) * rx * 0.06
            let x0 = c - half + shift
            let x1 = c + half + shift
            let slope = normal(&random) * 0.05
            let thick = big * uniform(&random, 0.006, 0.03) * (1 - 0.5 * abs(v))
            let wobble = uniform(&random, 0, big * 0.006)
            let phase = uniform(&random, 0, 6.3)
            let steps = 40
            var top: [CGPoint] = []
            var bottom: [CGPoint] = []
            for i in 0...steps {
                let t = CGFloat(i) / CGFloat(steps)
                let x = x0 + (x1 - x0) * t
                let center = y + slope * (x - c) + wobble * sin(phase + t * 9)
                let taper = min(1, t / 0.08, (1 - t) / 0.08)
                top.append(CGPoint(x: x, y: center - thick / 2 * taper))
                bottom.append(CGPoint(x: x, y: center + thick / 2 * taper))
            }
            context.addLines(between: top + bottom.reversed())
            context.closePath()
            context.fillPath()
        }
        return context.makeImage().map { CIImage(cgImage: $0) }
    }

    static func uniform(_ random: inout SplitMix, _ low: CGFloat, _ high: CGFloat) -> CGFloat {
        low + (high - low) * CGFloat(Double(random.next() >> 11) / Double(1 << 53))
    }

    // Box-Muller, mean 0, sigma 1.
    static func normal(_ random: inout SplitMix) -> CGFloat {
        let u = max(uniform(&random, 0, 1), 1e-9)
        let v = uniform(&random, 0, 1)
        return sqrt(-2 * log(u)) * cos(2 * .pi * v)
    }
}
