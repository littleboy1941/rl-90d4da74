import CoreImage
import Metal
import UIKit

// "Smear" (author's reference `docs/ref_smear.jpg` in fff, vertical glitch
// portrait): the face runs down in thin vertical streaks from the forehead
// to the chest or the bottom of the frame, the features only ghost through;
// a few streaks carry a faint cyan / magenta / yellow cast; the hair is
// dragged to the right in ragged horizontal lines; the whole frame is muted
// grey-green with lifted blacks. Streak parameters change column by column
// at staggered moments, so the picture lives without jumping as a whole.
// PC preview with the same math: `tools/sim_smear.py`.
struct SmearStyle: FaceStyle {
    let name = "smear"

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        []
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        return SmearGPU.shared.run(camera, face: face.bounds, time: time)
            ?? camera.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        image
    }
}

// Two compute passes: one thread per column drags each streak down the
// column (running average with a small per-column rate), then one thread
// per row drags the hair band to the right. Texture rows follow Core Image
// rows (bottom up), as in LeakGPU; the face box is in screen rows (top down).
final class SmearGPU {
    static let shared = SmearGPU()

    private struct Params {
        var width: Float
        var height: Float
        var faceX: Float                // face box, screen pixels (top down)
        var faceY: Float
        var faceW: Float
        var faceH: Float
        var time: Float
        var hasFace: UInt32
    }

    private let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private let columns: MTLComputePipelineState?
    private let rows: MTLComputePipelineState?
    private let context: CIContext?
    private var source: MTLTexture?
    private var middle: MTLTexture?
    private var target: MTLTexture?
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private init() {
        let metal = MTLCreateSystemDefaultDevice()
        device = metal
        queue = metal?.makeCommandQueue()
        context = metal.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
        let library = metal.flatMap { try? $0.makeLibrary(source: Self.shader, options: nil) }
        columns = library?.makeFunction(name: "smearColumns").flatMap { try? metal?.makeComputePipelineState(function: $0) }
        rows = library?.makeFunction(name: "smearRows").flatMap { try? metal?.makeComputePipelineState(function: $0) }
    }

    func run(_ image: CIImage, face: CGRect, time: Double) -> CIImage? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        guard let device, let queue, let columns, let rows, let context, width > 0, height > 0 else {
            return nil
        }
        if source?.width != width || source?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
            source = device.makeTexture(descriptor: descriptor)
            middle = device.makeTexture(descriptor: descriptor)
            target = device.makeTexture(descriptor: descriptor)
        }
        guard let source, let middle, let target,
              let commandBuffer = queue.makeCommandBuffer() else {
            return nil
        }
        context.render(image, to: source, commandBuffer: commandBuffer,
                       bounds: image.extent, colorSpace: colorSpace)
        let hasFace = !face.isNull && face.width > 4
        let box = hasFace ? face.offsetBy(dx: -image.extent.minX, dy: -image.extent.minY) : .zero
        var params = Params(
            width: Float(width), height: Float(height),
            faceX: Float(box.minX), faceY: Float(CGFloat(height) - box.maxY),
            faceW: Float(box.width), faceH: Float(box.height),
            time: Float(time.truncatingRemainder(dividingBy: 10_000)),
            hasFace: hasFace ? 1 : 0)
        for (pipeline, input, output, count) in [(columns, source, middle, width), (rows, middle, target, height)] {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(output, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: (count + 63) / 64, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encoder.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            return nil
        }
        return CIImage(mtlTexture: target, options: [.colorSpace: colorSpace])?
            .transformed(by: CGAffineTransform(translationX: image.extent.minX, y: image.extent.minY))
    }

    // Random values are hashed from the column (row) in a 540 px grid, so
    // the look does not depend on the frame size; rates are per 400 px, as
    // in the PC preview.
    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float width; float height;
        float faceX; float faceY; float faceW; float faceH;
        float time; uint hasFace;
    };

    uint hash32(uint x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    float unit2(uint a, uint b) {
        uint h = 0x9e3779b9u;
        h = hash32(a + h * 31u);
        h = hash32(b + h * 31u);
        return float(h & 0xffffu) / 65536.0;
    }

    float unit3(uint a, uint b, uint c) {
        uint h = 0x9e3779b9u;
        h = hash32(a + h * 31u);
        h = hash32(b + h * 31u);
        h = hash32(c + h * 31u);
        return float(h & 0xffffu) / 65536.0;
    }

    // Muted grey-green look: low saturation, lifted blacks.
    float3 grade(float3 c) {
        float l = dot(c, float3(0.299, 0.587, 0.114));
        c = l + (c - l) * 0.12;
        c = c * 0.86 + 0.07;
        return c * float3(0.97, 1.0, 0.98);
    }

    kernel void smearColumns(texture2d<float, access::read> cam [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             constant Params &prm [[buffer(0)]],
                             uint gid [[thread_position_in_grid]]) {
        uint w = cam.get_width();
        uint h = cam.get_height();
        if (gid >= w) {
            return;
        }
        float px = prm.width / 400.0;                 // preview pixel, in frame pixels
        uint x = uint(float(gid) * 540.0 / prm.width);
        float fy = prm.faceY;
        float fh = prm.faceH;
        float cx = prm.faceX + prm.faceW * 0.5;
        uint epoch = uint(floor(prm.time / 0.6 + unit2(x, 7u) * 3.0));
        uint band = x / 9u;
        float r[6];
        for (uint k = 0; k < 6; k++) {
            r[k] = 0.6 * unit3(x, epoch, k) + 0.4 * unit3(band, epoch, k + 10u);
        }
        float u = (float(gid) - cx) / max(prm.faceW * 0.55, 1.0);
        bool head = prm.hasFace != 0u && fabs(u) < 1.0 && r[0] < 0.93;
        bool ghost = prm.hasFace != 0u && !head && fabs(u) < 2.2 && r[0] < 0.06;
        float ellipse = 1.0 - sqrt(saturate(1.0 - u * u));
        float start = fy + fh * (-0.02 + 0.16 * r[1] + 0.3 * ellipse);
        float end = r[2] > 0.8 ? float(h) : fy + fh * (1.0 + 1.7 * pow(r[2], 0.7));
        float alpha = (0.003 + 0.09 * pow(r[3], 2.5)) / px;
        if (ghost) {
            start = fy + fh * r[1];
            end = start + fh * (0.5 + r[2]);
            alpha = 0.02 / px;
        }
        float3 tint = float3(1.0);
        if (r[4] < 0.08) {
            tint = float3(0.86, 1.04, 1.12);
        } else if (r[4] < 0.15) {
            tint = float3(1.1, 0.88, 1.08);
        } else if (r[4] < 0.21) {
            tint = float3(1.08, 1.06, 0.84);
        }
        float3 gain = (0.86 + 0.28 * r[5]) * tint;
        bool active = head || ghost;
        float3 acc = float3(0.0);
        bool started = false;
        for (uint y = 0; y < h; y++) {
            uint2 p = uint2(gid, h - 1u - y);
            float3 c = grade(cam.read(p).rgb);
            float sy = float(y);
            if (active && sy >= start && sy < end) {
                if (!started) {
                    acc = c;
                    started = true;
                }
                acc += (c - acc) * alpha;
                float3 streak = acc * gain;
                float fade = saturate((end - sy) / (6.0 * px));
                float3 m = c + (streak - c) * fade;
                if (ghost) {
                    m = c + (m - c) * 0.5;
                }
                c = m;
            }
            dst.write(float4(saturate(c), 1.0), p);
        }
    }

    kernel void smearRows(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant Params &prm [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
        uint w = src.get_width();
        uint h = src.get_height();
        if (gid >= h) {
            return;
        }
        float px = prm.width / 400.0;
        uint ty = h - 1u - gid;
        uint y = uint(float(gid) * 540.0 / prm.height);
        float sy = float(gid);
        float fy = prm.faceY;
        float fh = prm.faceH;
        float fw = prm.faceW;
        float cx = prm.faceX + fw * 0.5;
        bool on = prm.hasFace != 0u && sy >= fy - 0.55 * fh && sy < fy + 0.35 * fh;
        uint e = uint(floor(prm.time / 0.6 + unit2(y, 9u) * 3.0));
        if (0.6 * unit3(y, e, 20u) + 0.4 * unit3(y / 3u, e, 21u) > 0.6) {
            on = false;
        }
        float a = (0.03 + 0.17 * unit3(y, e, 22u)) / px;
        int x0 = int(max(0.0, cx - fw * 0.6));
        int x1 = int(min(float(w), cx + fw * (0.5 + 0.8 * unit3(y, e, 23u))));
        float3 acc = src.read(uint2(uint(min(x0, int(w) - 1)), ty)).rgb;
        for (uint x = 0; x < w; x++) {
            float3 c = src.read(uint2(x, ty)).rgb;
            if (on && int(x) >= x0 && int(x) < x1) {
                acc += (c - acc) * a;
                c = acc;
            }
            dst.write(float4(c, 1.0), uint2(x, ty));
        }
    }
    """
}
