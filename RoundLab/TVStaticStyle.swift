import CoreImage
import Metal
import UIKit

// "TV static" (author's reference `docs/ref_tvstatic.jpg` in fff, face
// only): the face (skin and features from the face-parsing network; hair,
// ears and the rest of the frame stay) is filled with TV snow - soft blobs
// stretched sideways with cyan / orange fringes, blown out to near white
// at the forehead and thinning into separate blobs towards the chin; a
// glow spills over the face edge; faint vertical aperture-grille lines.
// The snow is new every frame (it boils, as on a TV with no signal) with
// a slight flicker. PC preview: `tools/sim_tvstatic.py`.
struct TVStaticStyle: FaceStyle {
    let name = "tv_static"
    var glow: Double = 0.55           // glow strength
    var glowRadius: CGFloat = 0.05    // glow sigma, of the face width

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        []
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func compose(camera: CIImage, overlay: CIImage, face: FaceShape, time: Double) -> CIImage {
        let extent = camera.extent
        let bounds = face.bounds
        guard let mask = FaceParser.shared.faceMask(camera, around: bounds, classes: FaceParser.skinClasses) else {
            // No network: never show the bare face.
            return camera.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
        guard let snow = TVStaticGPU.shared.run(size: extent.size, face: bounds, time: time)?
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY)) else {
            return camera.applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: 24]).cropped(to: extent)
        }
        let soft = mask.clampedToExtent()
            .applyingFilter("CIMorphologyMaximum", parameters: [
                kCIInputRadiusKey: bounds.isNull ? 2 : max(2, bounds.width * 0.015),
            ])
            .applyingGaussianBlur(sigma: 1.0)
            .cropped(to: extent)
        let black = CIImage(color: .black).cropped(to: extent)
        let masked = snow.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: black,
            kCIInputMaskImageKey: soft,
        ])
        let halo = masked.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(max((bounds.isNull ? 400 : bounds.width) * glowRadius, 2)))
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: CGFloat(glow) * 0.9, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: CGFloat(glow) * 0.95, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(glow), w: 0),
            ])
            .cropped(to: extent)
        return snow.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: camera,
            kCIInputMaskImageKey: soft,
        ])
        .applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: halo])
        .cropped(to: extent)
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        image
    }
}

// One compute pass that draws the snow over the whole frame (the mask
// keeps only the face). Value noise on a grid of 4 x 2 px cells of a
// 400 px frame (scaled to the real size), reseeded every frame; red and
// blue read 2 px apart for the fringes; exposure by the face box.
// Texture rows follow Core Image rows (bottom up); the face box is in
// screen rows (top down).
final class TVStaticGPU {
    static let shared = TVStaticGPU()

    private struct Params {
        var width: Float
        var height: Float
        var faceY: Float                // face box top and height, screen pixels
        var faceH: Float
        var seed: UInt32
        var flicker: Float
    }

    private let queue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?
    private var target: MTLTexture?
    private var frame: UInt32 = 0
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    private init() {
        let metal = MTLCreateSystemDefaultDevice()
        queue = metal?.makeCommandQueue()
        let function = metal.flatMap { try? $0.makeLibrary(source: Self.shader, options: nil) }?
            .makeFunction(name: "snow")
        pipeline = function.flatMap { try? metal?.makeComputePipelineState(function: $0) }
    }

    func run(size: CGSize, face: CGRect, time: Double) -> CIImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        guard let queue, let pipeline, width > 0, height > 0 else {
            return nil
        }
        if target?.width != width || target?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            target = queue.device.makeTexture(descriptor: descriptor)
        }
        guard let target, let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        frame &+= 1
        let hasFace = !face.isNull && face.height > 4
        var params = Params(
            width: Float(width), height: Float(height),
            faceY: hasFace ? Float(size.height - face.maxY) : 0,
            faceH: hasFace ? Float(face.height) : Float(height),
            seed: frame &* 2_654_435_761,
            flicker: 1 + 0.04 * Float(sin(time * 23) * cos(time * 7)))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(target, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
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

    struct Params {
        float width; float height;
        float faceY; float faceH;
        uint seed; float flicker;
    };

    uint hash32(uint x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    float cell(int x, int y, uint seed) {
        uint h = hash32(uint(x) * 73856093u ^ uint(y) * 19349663u ^ seed);
        return float(h & 0xffffu) / 65535.0;
    }

    // Smooth value noise: 4 x 2 px cells of a 400 px frame.
    float noise(float2 p, float scale, uint seed) {
        float2 q = p / (float2(4.0, 2.0) * scale);
        int2 i = int2(floor(q));
        float2 f = q - floor(q);
        f = f * f * (3.0 - 2.0 * f);
        float a = cell(i.x, i.y, seed), b = cell(i.x + 1, i.y, seed);
        float c = cell(i.x, i.y + 1, seed), d = cell(i.x + 1, i.y + 1, seed);
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    kernel void snow(texture2d<float, access::write> dst [[texture(0)]],
                     constant Params &prm [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
            return;
        }
        float scale = prm.width / 400.0;
        float2 p = float2(float(gid.x), prm.height - 1.0 - float(gid.y));
        // Forehead (top of the face box) blown out, thinning to the chin.
        float t = saturate((p.y - prm.faceY) / max(prm.faceH, 1.0));
        float gain = (2.8 - 2.0 * t) * prm.flicker;
        float cut = 0.05 + 0.45 * t;
        float3 c;
        for (int k = 0; k < 3; k++) {
            float dx = float(1 - k) * 2.0 * scale;          // red +2, green 0, blue -2
            float s = saturate((noise(p + float2(dx, 0.0), scale, prm.seed) - 0.5) * 1.6 + 0.5);
            c[k] = saturate((s - cut) / (1.0 - cut)) * gain;
        }
        c *= float3(1.0, 0.93, 0.97);                        // warm white
        c *= 0.88 + 0.12 * cos(p.x * 6.2831853 / (2.4 * scale));
        dst.write(float4(saturate(c), 1.0), gid);
    }
    """
}
