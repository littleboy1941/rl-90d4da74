import ARKit
import CoreImage
import Metal
import SceneKit
import UIKit

// "ASCII face" (author 2026-10-22): only the face - the ARKit mesh, forehead
// to chin, plus the skin found around it in the same frame (the mesh lags
// fast head turns) - is drawn with real everyday Linux command lines on
// black; the rest of the frame stays the live camera. Light cells show whole readable commands,
// darker cells drop letters, and on the face contours only the command
// syntax that matches the edge direction stays (`/` from paths, `|` from
// pipes, `-` from flags, `\`). Commands that change or delete things
// (rm, sudo) are tinted red. Two palettes: Mr. Robot (cold terminal white)
// and Lain (bleached light, dark red characters in the shadows).
// One Metal pass over 5x8 px cells; preview: tools/sim_ascii.py.
struct AsciiStyle: FaceStyle {
    enum Palette: UInt32 {
        case robot = 0
        case lain = 1
    }

    let palette: Palette
    var scrollPerSecond: Float = 6           // characters per second
    var meshScale: Float = 1.08              // mask mesh around the head origin
    var maskGrow: CGFloat = 0.06             // mask dilation, of the face width

    var name: String {
        palette == .robot ? "ascii_robot" : "ascii_lain"
    }

    // White unlit face mesh (eyes and mouth closed): the mask of the face.
    func makeNodes(device: MTLDevice) -> [StyledNode] {
        guard let mesh = ARSCNFaceGeometry(device: device, fillMesh: true) else {
            return []
        }
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.white
        material.lightingModel = .constant
        mesh.firstMaterial = material
        // The ARKit mesh stops short of the cheek sides, temples and under the
        // chin; a slightly larger mesh reaches them.
        return [StyledNode(node: SCNNode(geometry: mesh), scale: meshScale)]
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        guard let result = AsciiGPU.shared.run(camera, face: face.bounds, palette: palette,
                                               scroll: Float(time) * scrollPerSecond) else {
            // Shader missing (compile error on the device): never show the bare face.
            return camera.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
        let text = result.text
        // Two masks, whichever covers more: the ARKit mesh (grown outward;
        // it lags behind fast head turns and stops short of the cheek sides)
        // and the skin found in this very frame around the face, so the text
        // stays on the face while it moves. Edge softened by ~1.5 px.
        let bounds = face.bounds
        let grow = bounds.isNull ? 0 : bounds.width * maskGrow
        let mesh = overlay.clampedToExtent()
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: grow])
        let skin = result.skin.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 1, y: 0, z: 0, w: 0),
        ]).clampedToExtent()
        let mask = mesh.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: skin])
            .applyingGaussianBlur(sigma: 1.5)
            .cropped(to: extent)
        return text.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: camera,
            kCIInputMaskImageKey: mask,
        ]).cropped(to: extent)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        image
    }

    // Everyday commands: files, git, builds, packages, media, system status.
    static let commands = [
        "ls -la ~/projects",
        "cd ~/projects/roundlab && git status",
        "git log --oneline --graph | head -20",
        "git add -A && git commit -m 'fix frame timing'",
        "make -j8 && ./build/roundlab --verbose",
        "find . -name '*.swift' | xargs wc -l | sort -n",
        "grep -rn TODO src/ | less",
        "du -sh /var/log/* | sort -h",
        "df -h /home",
        "tar -czvf backup_2026.tar.gz ~/documents",
        "rsync -av ~/photos/ /mnt/backup/photos/",
        "ffmpeg -i round.mov -vf scale=400:400 round.mp4",
        "sudo apt update && sudo apt upgrade -y",
        "systemctl status nginx --no-pager",
        "journalctl -u nginx --since today | tail -50",
        "ps aux | grep python | grep -v grep",
        "uname -a; uptime; free -h",
        "cat /proc/cpuinfo | grep 'model name' | uniq",
        "chmod +x scripts/deploy.sh",
        "python3 -m venv .venv && source .venv/bin/activate",
        "pip install -r requirements.txt",
        "docker ps --format '{{.Names}}\\t{{.Status}}'",
        "rm -rf build/ && mkdir build",
        "echo $PATH | tr ':' '\\n'",
        "sed -i 's/old_name/new_name/g' config.ini",
        "awk -F, '{print $2}' data.csv | sort | uniq -c",
        "watch -n 1 nvidia-smi",
        "man bash",
    ]

    // ASCII codes of all commands joined by three spaces; bit 7 marks the
    // characters of tinted commands.
    static let stream: [UInt8] = {
        var bytes: [UInt8] = []
        for command in commands {
            let tinted = command.hasPrefix("rm ") || command.hasPrefix("sudo ")
            for scalar in command.unicodeScalars where scalar.isASCII {
                bytes.append(UInt8(scalar.value) | (tinted ? 0x80 : 0))
            }
            bytes += [32, 32, 32]
        }
        return bytes
    }()
}

// Glyph atlas (printable ASCII 32...126, one 5x8 cell each, white on black)
// plus one compute pass that draws the cells.
final class AsciiGPU {
    static let shared = AsciiGPU()
    static let cellWidth = 5
    static let cellHeight = 8

    private struct Params {
        var width: Float
        var height: Float
        var cellW: Float
        var cellH: Float
        var faceX: Float                // face box, screen pixels (top down)
        var faceY: Float
        var faceW: Float
        var faceH: Float
        var scroll: Float
        var streamLength: UInt32
        var palette: UInt32
    }

    private let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?
    private let context: CIContext?
    private let atlas: MTLTexture?
    private let stream: MTLBuffer?
    private var source: MTLTexture?
    private var target: MTLTexture?
    private var skin: MTLTexture?
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private init() {
        let metal = MTLCreateSystemDefaultDevice()
        device = metal
        queue = device?.makeCommandQueue()
        context = device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
        if let device,
           let library = try? device.makeLibrary(source: Self.shader, options: nil),
           let function = library.makeFunction(name: "ascii") {
            pipeline = try? device.makeComputePipelineState(function: function)
        } else {
            pipeline = nil
        }
        atlas = metal.flatMap(Self.makeAtlas)
        stream = AsciiStyle.stream.withUnsafeBytes { bytes in
            metal?.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
    }

    // Menlo Bold, advance ~5 px at 8.3 pt; row 0 of the texture is the top.
    private static func makeAtlas(device: MTLDevice) -> MTLTexture? {
        let count = 95
        let width = count * cellWidth
        let height = cellHeight
        let font = UIFont(name: "Menlo-Bold", size: 8.3) ?? .monospacedSystemFont(ofSize: 8.3, weight: .bold)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let image = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format).image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            for index in 0..<count {
                let text = String(UnicodeScalar(UInt8(32 + index)))
                (text as NSString).draw(
                    at: CGPoint(x: CGFloat(index * cellWidth), y: -1),
                    withAttributes: [.font: font, .foregroundColor: UIColor.white])
            }
        }
        guard let cgImage = image.cgImage,
              let gray = CGContext(
                  data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let data = gray.data else {
            return nil
        }
        gray.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0, withBytes: data, bytesPerRow: width)
        return texture
    }

    // Text over black, plus the per-cell skin mask (red channel, 0 or 1).
    func run(_ image: CIImage, face: CGRect, palette: AsciiStyle.Palette, scroll: Float)
        -> (text: CIImage, skin: CIImage)? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        guard let device, let queue, let pipeline, let context, let atlas, let stream,
              width > 0, height > 0 else {
            return nil
        }
        if source?.width != width || source?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
            source = device.makeTexture(descriptor: descriptor)
            target = device.makeTexture(descriptor: descriptor)
            let mask = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
            mask.storageMode = .private
            mask.usage = [.shaderRead, .shaderWrite]
            skin = device.makeTexture(descriptor: mask)
        }
        guard let source, let target, let skin,
              let commandBuffer = queue.makeCommandBuffer() else {
            return nil
        }
        context.render(image, to: source, commandBuffer: commandBuffer,
                       bounds: image.extent, colorSpace: colorSpace)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        let box = face.isNull ? CGRect.zero : face
        var params = Params(
            width: Float(width), height: Float(height),
            cellW: Float(Self.cellWidth), cellH: Float(Self.cellHeight),
            faceX: Float(box.minX), faceY: Float(CGFloat(height) - box.maxY),
            faceW: Float(box.width), faceH: Float(box.height),
            scroll: scroll.truncatingRemainder(dividingBy: 100_000),
            streamLength: UInt32(AsciiStyle.stream.count),
            palette: palette.rawValue)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(atlas, index: 1)
        encoder.setTexture(target, index: 2)
        encoder.setTexture(skin, index: 3)
        encoder.setBuffer(stream, offset: 0, index: 0)
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
        guard let text = CIImage(mtlTexture: target, options: [.colorSpace: colorSpace]),
              let mask = CIImage(mtlTexture: skin, options: [.colorSpace: NSNull()]) else {
            return nil
        }
        return (text, mask)
    }

    // Texture rows follow Core Image rows (bottom up), as in LeakGPU; the
    // cell grid and the face box are in screen rows (top down).
    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { float width; float height; float cellW; float cellH;
                    float faceX; float faceY; float faceW; float faceH;
                    float scroll; uint streamLength; uint palette; };

    // Luma at a screen point (x, y top down).
    static float luma(texture2d<float, access::read> cam, float2 s, constant Params &prm) {
        int x = clamp(int(s.x), 0, int(prm.width) - 1);
        int y = clamp(int(prm.height - 1.0 - s.y), 0, int(prm.height) - 1);
        return dot(cam.read(uint2(x, y)).rgb, float3(0.299, 0.587, 0.114));
    }

    static float3 rgbAt(texture2d<float, access::read> cam, float2 s, constant Params &prm) {
        int x = clamp(int(s.x), 0, int(prm.width) - 1);
        int y = clamp(int(prm.height - 1.0 - s.y), 0, int(prm.height) - 1);
        return cam.read(uint2(x, y)).rgb;
    }

    // Classic YCbCr skin box (Cr 135-180, Cb 85-135 of 255), not too dark.
    static bool isSkin(float3 c) {
        float y = dot(c, float3(0.299, 0.587, 0.114));
        float cb = 0.5 - 0.1687 * c.r - 0.3313 * c.g + 0.5 * c.b;
        float cr = 0.5 + 0.5 * c.r - 0.4187 * c.g - 0.0813 * c.b;
        return y > 0.15 && cr > 0.53 && cr < 0.71 && cb > 0.33 && cb < 0.53;
    }

    static uint edgeGlyph(float2 g) {
        // g = brightness gradient (screen x right, y down); the edge runs across it.
        float a = fmod(atan2(g.y, g.x) * 57.29578 + 360.0, 180.0);
        if (a < 22.5 || a >= 157.5) { return uint('|'); }
        if (a < 67.5) { return uint('/'); }
        if (a < 112.5) { return uint('-'); }
        return uint('\\\\');
    }

    kernel void ascii(texture2d<float, access::read> cam [[texture(0)]],
                      texture2d<float, access::read> atlas [[texture(1)]],
                      texture2d<float, access::write> dst [[texture(2)]],
                      texture2d<float, access::write> skinOut [[texture(3)]],
                      constant uchar *stream [[buffer(0)]],
                      constant Params &prm [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
            return;
        }
        float sy = prm.height - 1.0 - float(gid.y);
        uint col = uint(float(gid.x) / prm.cellW);
        uint row = uint(sy / prm.cellH);
        float2 center = float2((float(col) + 0.5) * prm.cellW, (float(row) + 0.5) * prm.cellH);

        // Cells around the face box are drawn (the box lags fast turns, so it
        // is generous); the final shape is the mesh and skin masks.
        float fade = 0.0;
        float skin = 0.0;
        if (prm.faceW > 4.0) {
            float2 lo = float2(prm.faceX - prm.faceW * 0.25, prm.faceY - prm.faceH * 0.15);
            float2 hi = float2(prm.faceX + prm.faceW * 1.25, prm.faceY + prm.faceH * 1.08);
            fade = all(center >= lo) && all(center <= hi) ? 1.0 : 0.0;
            if (fade > 0.0) {
                // Skin cell: at least 3 of 4 taps look like skin in this frame.
                int hits = int(isSkin(rgbAt(cam, center + float2(-1, -2), prm)))
                         + int(isSkin(rgbAt(cam, center + float2(1, -2), prm)))
                         + int(isSkin(rgbAt(cam, center + float2(-1, 2), prm)))
                         + int(isSkin(rgbAt(cam, center + float2(1, 2), prm)));
                skin = hits >= 3 ? 1.0 : 0.0;
            }
        }
        skinOut.write(float4(skin, 0.0, 0.0, 1.0), gid);
        float3 color = float3(0.0);
        if (fade > 0.0) {
            // Cell brightness (4 taps) against its surroundings (8 taps, 12 px):
            // local contrast so eyes and mouth separate from even skin.
            float l = 0.25 * (luma(cam, center + float2(-1, -2), prm) + luma(cam, center + float2(1, -2), prm)
                            + luma(cam, center + float2(-1, 2), prm) + luma(cam, center + float2(1, 2), prm));
            float m = 0.0;
            for (int k = 0; k < 8; k++) {
                float a = float(k) * 0.785398;
                m += luma(cam, center + 12.0 * float2(cos(a), sin(a)), prm);
            }
            m /= 8.0;
            float lc = saturate(l + (l - m) * 1.5);
            float2 g = float2(luma(cam, center + float2(3, 0), prm) - luma(cam, center - float2(3, 0), prm),
                              luma(cam, center + float2(0, 3), prm) - luma(cam, center - float2(0, 3), prm));
            bool edge = length(g) > 0.14;

            uint k = (row * 37u + uint(prm.scroll) + col) % prm.streamLength;
            uint code = uint(stream[k]);
            bool tinted = (code & 0x80u) != 0u;
            code &= 0x7fu;
            if (edge) {
                code = edgeGlyph(g);
            }
            float v = pow(smoothstep(0.3, 0.8, lc), 1.8);
            if (edge) {
                v = max(v, 0.35);
            }
            v *= fade;
            float3 ink = float3(0.0);
            bool draw = code > 32u && code < 127u;
            if (prm.palette == 1u) {
                if (lc < 0.3 && !edge) {
                    draw = draw && ((row + col) % 2u == 1u);
                    ink = float3(0.55, 0.03, 0.08) * fade;      // shadows: dark red
                } else {
                    float gv = (60.0 + 195.0 * v) / 255.0;
                    ink = float3(gv, gv * 0.94, gv * 0.96);
                }
            } else {
                draw = draw && v >= 0.06;
                if (tinted && !edge) {
                    ink = float3(0.9 * max(v, 0.5), 0.11, 0.15);
                } else {
                    ink = float3(v * 0.86, v * 0.95, v);
                }
            }
            if (draw) {
                uint lx = uint(float(gid.x) - float(col) * prm.cellW);
                uint ly = uint(sy - float(row) * prm.cellH);
                float glyph = atlas.read(uint2((code - 32u) * uint(prm.cellW) + lx, ly)).r;
                color = ink * glyph;
            }
        }
        dst.write(float4(color, 1.0), gid);
    }
    """
}
