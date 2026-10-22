import ARKit
import CoreImage
import Metal
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
// Streaks, fringe and LED matrix run in one Metal pass (LeakGPU); the
// program is the same every frame, only the per-row numbers change.
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
        let image = bright.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: dark,
            kCIInputMaskImageKey: soft,
        ]).cropped(to: extent)

        let rows = streakRows(face: face, extent: extent, time: time)
        let leaked = LeakGPU.shared.run(image, rows: rows, cell: Float(cell))
            ?? SignalEffect.colorSplit(image, extent: extent, shift: 2)
        return finish(leaked, time: time)
    }

    // Per LED row (Core Image rows, bottom up): streak and slide numbers.
    func streakRows(face: FaceShape, extent: CGRect, time: Double) -> [LeakRow] {
        let count = Int(ceil(extent.height / cell))
        var rows = [LeakRow](repeating: LeakRow(), count: count)
        let bounds = face.bounds
        guard !bounds.isNull, bounds.width > 4 else {
            return rows
        }
        var left = [CGFloat](repeating: .greatestFiniteMagnitude, count: count)
        var right = [CGFloat](repeating: -.greatestFiniteMagnitude, count: count)
        for point in face.points {
            let row = Int(point.y / cell)
            if row >= 0, row < count {
                left[row] = min(left[row], point.x)
                right[row] = max(right[row], point.x)
            }
        }
        let tick = UInt64(max(0, Int(time * ticksPerSecond)))
        for row in 0..<count where left[row] < .greatestFiniteMagnitude {
            var generator = SplitMix(seed: UInt64(row) &* 7919 &+ tick &* 104_729)
            let roll = Double(generator.next() % 10_000) / 10_000
            var params = LeakRow()
            params.left = Float(left[row])
            params.right = Float(right[row])
            if roll < shiftedShare {
                params.shift = -Float(cell) * Float(2 + generator.next() % 6)
            }
            if roll >= emptyShare {
                let strength = (roll - emptyShare) / (1 - emptyShare)
                let inside = bounds.width * (0.04 + 0.1 * Double(generator.next() % 100) / 100)
                params.edge = Float(left[row])
                params.length = Float(bounds.width * maxLength * CGFloat(0.2 + 0.8 * strength))
                params.sampleX = Float(min(left[row] + inside, extent.maxX - 1))
            }
            rows[row] = params
        }
        return rows
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        let extent = image.extent
        var result = image.applyingFilter("CIVignette", parameters: [
            kCIInputIntensityKey: 0.7,
            kCIInputRadiusKey: 1.5,
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

// "Mr. Swirl": the Photoshop Twirl look from the 2007 Interpol case - a
// strong whirlpool centered on the face, the rest of the frame untouched.
// Core Image's twirl keeps the same graph every frame; only the center,
// radius and angle change. A plain twirl can be undone (that is how the
// 2007 photos were unswirled), so this is a look, not anonymization.
struct SwirlStyle: FaceStyle {
    let name = "mr_swirl"
    var turns: Double = 1.6           // full turns at the center
    var reach: CGFloat = 0.62         // radius as a share of the larger face side

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        []
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        let bounds = face.bounds
        guard !bounds.isNull, bounds.width > 4 else {
            return camera
        }
        let swirled = camera.clampedToExtent().applyingFilter("CITwirlDistortion", parameters: [
            kCIInputCenterKey: CIVector(x: bounds.midX, y: bounds.midY),
            kCIInputRadiusKey: max(bounds.width, bounds.height) * reach,
            kCIInputAngleKey: turns * 2 * Double.pi,
        ])
        return finish(swirled.cropped(to: extent), time: time)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        image
    }
}

// "Melt" (author's reference 4, datamoshed anime still): the picture runs
// down in vertical bands of block width, like repeated P-frames with a
// downward motion vector. Under each band's start line every pixel copies
// the previous output from a few pixels above, so the line above keeps
// streaming down; now and then a whole band-wide chunk (16 px tall)
// refreshes from the camera and falls too (block steps inside the drips). Fresh pixels get block chroma
// (8 px), a mauve grade and rare pink / acid-green chroma errors; dragged
// pixels pick up speckle that streaks into fine vertical lines. Bands over
// the face and hair melt from the forehead / eyes down; the background
// melts less. Every few seconds a keyframe resets all drips.
struct MeltStyle: FaceStyle {
    let name = "melt"
    var block: Float = 4                // block size in output pixels
    var epochSeconds = 3.5              // keyframe + new bands
    var faceMeltShare = 0.85            // bands over the face that melt
    var backgroundMeltShare = 0.4

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        []
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        let epoch = Int(time / epochSeconds)
        let columns = meltColumns(face: face.bounds, extent: extent, epoch: epoch)
        return MeltGPU.shared.run(camera, columns: columns, block: block, epoch: epoch, time: time) ?? camera
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        image
    }

    // Bands of 1-6 blocks; each band's start line (screen y, top down) and
    // speed in pixels per frame. Same bands for the whole epoch.
    func meltColumns(face: CGRect, extent: CGRect, epoch: Int) -> [MeltColumn] {
        let count = Int(ceil(Float(extent.width) / block))
        var columns = [MeltColumn](repeating: MeltColumn(), count: count)
        var generator = SplitMix(seed: UInt64(max(epoch, 0)) &* 7919 &+ 17)
        let height = Float(extent.height)
        let hasFace = !face.isNull && face.width > 4
        // Face box widened for the hair; screen y grows downward.
        let hairLeft = Float(face.minX - face.width * 0.3)
        let hairRight = Float(face.maxX + face.width * 0.3)
        let faceTop = height - Float(face.maxY)
        let faceHeight = Float(face.height)
        func unit() -> Float {
            Float(generator.next() % 10_000) / 10_000
        }
        var index = 0
        while index < count {
            let width = 1 + Int(generator.next() % 6)
            let center = (Float(index) + Float(width) / 2) * block
            let overFace = hasFace && center > hairLeft && center < hairRight
            var column = MeltColumn()
            if overFace {
                if Double(unit()) < faceMeltShare {
                    column.top = faceTop - 0.15 * faceHeight + unit() * 0.6 * faceHeight
                    column.speed = Float(1 + generator.next() % 4)
                    column.refresh = 0.006
                }
            } else if Double(unit()) < backgroundMeltShare {
                column.top = height * (0.15 + 0.75 * unit())
                column.speed = Float(1 + generator.next() % 3)
                column.refresh = 0.01
            }
            column.band = Float(index)
            for i in index..<min(index + width, count) {
                columns[i] = column
            }
            index += width
        }
        return columns
    }
}

// Same memory layout as `Column` in MeltGPU's Metal source (4 floats).
struct MeltColumn {
    var top: Float = 0
    var speed: Float = 0
    var refresh: Float = 1
    var band: Float = 0               // first block column of the band
}

// One compute pass with feedback: reads the camera and the previous output,
// writes the next output. Three output textures in a ring, so the one Core
// Image is still showing is never the one being written. Texture rows
// follow Core Image rows (bottom up), as in LeakGPU.
final class MeltGPU {
    static let shared = MeltGPU()

    private struct Params {
        var width: Float
        var height: Float
        var block: Float
        var columns: UInt32
        var frame: UInt32
        var keyframe: UInt32
    }

    private let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?
    private let context: CIContext?
    private var source: MTLTexture?
    private var ring: [MTLTexture] = []
    private var current = 0
    private var frame: UInt32 = 0
    private var lastEpoch = -1
    private var lastTime = 0.0
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private init() {
        device = MTLCreateSystemDefaultDevice()
        queue = device?.makeCommandQueue()
        context = device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
        if let device,
           let library = try? device.makeLibrary(source: Self.shader, options: nil),
           let function = library.makeFunction(name: "melt") {
            pipeline = try? device.makeComputePipelineState(function: function)
        } else {
            pipeline = nil
        }
    }

    func run(_ image: CIImage, columns: [MeltColumn], block: Float, epoch: Int, time: Double) -> CIImage? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        guard let device, let queue, let pipeline, let context, !columns.isEmpty, width > 0, height > 0 else {
            return nil
        }
        var keyframe = epoch != lastEpoch || time - lastTime > 0.25
        if source?.width != width || source?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
            source = device.makeTexture(descriptor: descriptor)
            ring = (0..<3).compactMap { _ in device.makeTexture(descriptor: descriptor) }
            keyframe = true
        }
        lastEpoch = epoch
        lastTime = time
        guard let source, ring.count == 3,
              let commandBuffer = queue.makeCommandBuffer() else {
            return nil
        }
        let previous = ring[current]
        current = (current + 1) % 3
        let target = ring[current]
        frame &+= 1
        let graded = image.applyingFilter("CIColorControls", parameters: [
            kCIInputSaturationKey: 0.55,
            kCIInputContrastKey: 1.12,
        ])
        context.render(
            graded,
            to: source,
            commandBuffer: commandBuffer,
            bounds: image.extent,
            colorSpace: colorSpace)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        var params = Params(
            width: Float(width), height: Float(height), block: block,
            columns: UInt32(columns.count), frame: frame, keyframe: keyframe ? 1 : 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(previous, index: 1)
        encoder.setTexture(target, index: 2)
        columns.withUnsafeBytes { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 0)
        }
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            return nil
        }
        return CIImage(mtlTexture: target, options: [.colorSpace: colorSpace])
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct Column { float top; float speed; float refresh; float band; };
    struct Params { float width; float height; float block; uint columns; uint frame; uint keyframe; };

    static float hash3(uint a, uint b, uint c) {
        uint h = a * 73856093u ^ b * 19349663u ^ c * 83492791u;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        return float(h & 0xffffffu) / 16777216.0;
    }

    // Fresh pixel: luma from the pixel, chroma from its 2x2-block cell,
    // mauve grade, rare chroma errors that stick to the cell for a while.
    static float3 fresh(texture2d<float, access::read> cam, uint2 p, uint bx, uint by,
                        constant Params &prm) {
        int w = int(prm.width);
        int h = int(prm.height);
        float cell = prm.block * 2.0;
        int2 q = int2(floor(float2(p) / cell) * cell + cell * 0.5);
        q = clamp(q, int2(0), int2(w - 1, h - 1));
        float3 rgb = cam.read(p).rgb;
        float3 chroma = cam.read(uint2(q)).rgb;
        float y = dot(rgb, float3(0.299, 0.587, 0.114));
        float cy = dot(chroma, float3(0.299, 0.587, 0.114));
        float cb = (chroma.b - cy) * 0.564 * 0.8 + 0.02;
        float cr = (chroma.r - cy) * 0.713 * 0.8 + 0.045;
        float e = hash3(bx / 2u, by / 2u, prm.frame / 20u);
        if (e > 0.9975) { cb = 0.06; cr = 0.2; }            // pink
        else if (e > 0.995) { cb = -0.22; cr = -0.08; }     // acid green
        y = y * 0.92 + 0.05;
        return saturate(float3(y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb));
    }

    kernel void melt(texture2d<float, access::read> cam [[texture(0)]],
                     texture2d<float, access::read> prev [[texture(1)]],
                     texture2d<float, access::write> dst [[texture(2)]],
                     constant Column *cols [[buffer(0)]],
                     constant Params &prm [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
            return;
        }
        int h = int(prm.height);
        float sy = prm.height - 1.0 - float(gid.y);           // screen y, top down
        uint bx = uint(float(gid.x) / prm.block);
        uint by = uint(sy / prm.block);
        Column col = cols[min(bx, prm.columns - 1u)];
        bool zone = col.speed > 0.0 && sy > col.top;
        bool refresh = prm.keyframe != 0u || !zone
            || hash3(uint(col.band), by / 4u, prm.frame) < col.refresh;
        float3 c;
        if (refresh) {
            c = fresh(cam, gid, bx, by, prm);
        } else {
            int py = min(int(gid.y) + int(col.speed), h - 1);  // from above on screen
            c = prev.read(uint2(gid.x, uint(py))).rgb;
            float n = hash3(gid.x, gid.y, prm.frame);
            if (n > 0.997) {
                c = mix(c, float3(0.95, 0.78, 0.88), 0.5);
            } else if (n < 0.002) {
                c *= 0.8;
            }
        }
        dst.write(float4(c, 1.0), gid);
    }
    """
}

// Same memory layout as `Row` in LeakGPU's Metal source (6 floats).
struct LeakRow {
    var edge: Float = 0
    var length: Float = 0
    var sampleX: Float = 0
    var left: Float = 0
    var right: Float = 0
    var shift: Float = 0
}

// One compute pass: streaks + slid rows + color fringe + LED dot matrix.
// Texture rows follow Core Image rows (bottom up) in both directions, so
// row indices match LeakRow indices without flipping.
final class LeakGPU {
    static let shared = LeakGPU()

    private struct Params {
        var cell: Float
        var width: Float
        var height: Float
        var rows: UInt32
    }

    private let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?
    private let context: CIContext?
    private var source: MTLTexture?
    private var target: MTLTexture?
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private init() {
        device = MTLCreateSystemDefaultDevice()
        queue = device?.makeCommandQueue()
        context = device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
        if let device,
           let library = try? device.makeLibrary(source: Self.shader, options: nil),
           let function = library.makeFunction(name: "leak") {
            pipeline = try? device.makeComputePipelineState(function: function)
        } else {
            pipeline = nil
        }
    }

    func run(_ image: CIImage, rows: [LeakRow], cell: Float) -> CIImage? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        guard let device, let queue, let pipeline, let context, !rows.isEmpty, width > 0, height > 0 else {
            return nil
        }
        if source?.width != width || source?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
            source = device.makeTexture(descriptor: descriptor)
            target = device.makeTexture(descriptor: descriptor)
        }
        guard let source, let target,
              let commandBuffer = queue.makeCommandBuffer() else {
            return nil
        }
        context.render(
            image,
            to: source,
            commandBuffer: commandBuffer,
            bounds: image.extent,
            colorSpace: colorSpace)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        var params = Params(cell: cell, width: Float(width), height: Float(height), rows: UInt32(rows.count))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(target, index: 1)
        rows.withUnsafeBytes { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 0)
        }
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            return nil
        }
        return CIImage(mtlTexture: target, options: [.colorSpace: colorSpace])
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct Row { float edge; float length; float sampleX; float left; float right; float shift; };
    struct Params { float cell; float width; float height; uint rows; };

    static float4 leaked(texture2d<float, access::read> src, int2 p,
                         constant Row *rows, constant Params &prm) {
        int w = int(prm.width);
        int h = int(prm.height);
        p = clamp(p, int2(0), int2(w - 1, h - 1));
        uint index = min(uint(float(p.y) / prm.cell), prm.rows - 1);
        Row row = rows[index];
        float x = float(p.x);
        float4 c = src.read(uint2(p));
        if (row.shift != 0.0 && x >= row.left && x <= row.right) {
            int sx = clamp(int(x - row.shift), 0, w - 1);
            c = src.read(uint2(sx, p.y));
        }
        if (row.length > 0.0 && x >= row.edge - row.length && x < row.sampleX) {
            c = src.read(uint2(int(row.sampleX), p.y));
            if (x < row.edge - row.length * 0.65) {
                c.rgb *= float3(0.35, 0.8, 0.75);
            }
        }
        return c;
    }

    kernel void leak(texture2d<float, access::read> src [[texture(0)]],
                     texture2d<float, access::write> dst [[texture(1)]],
                     constant Row *rows [[buffer(0)]],
                     constant Params &prm [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
            return;
        }
        float cell = prm.cell;
        float2 origin = floor(float2(gid) / cell) * cell;
        int2 center = int2(origin + cell * 0.5);
        float r = leaked(src, center + int2(2, 0), rows, prm).r;
        float g = leaked(src, center, rows, prm).g;
        float b = leaked(src, center - int2(2, 0), rows, prm).b;
        float2 local = float2(gid) - origin + 0.5;
        float half_size = (cell - 1.5) * 0.5;
        float radius = cell * 0.2;
        float2 q = abs(local - (0.5 + half_size)) - (half_size - radius);
        float d = length(max(q, 0.0)) - radius;
        float led = d <= 0.0 ? 1.0 : 0.12;
        dst.write(float4(float3(r, g, b) * led, 1.0), gid);
    }
    """
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
