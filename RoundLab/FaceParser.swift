import CoreImage
import CoreML
import CoreVideo
import QuartzCore

// Per-frame face mask from a face-parsing network (YOLO26-nano, 19 classes,
// CelebAMask-HQ; Models/FaceParsing.mlpackage, compiled into the app by CI).
// Unlike the ARKit mesh, the mask comes from the very frame it is applied
// to, so it does not lag behind fast head turns. Default mask = the whole
// head: face parts plus ears, hair, hat (no neck, clothes) - the edge then
// runs along the head outline against the background, which the network
// sees far more surely than the forehead/hair line; `skinClasses` = the
// face alone (tv_static). Soft edge from the summed
// class probabilities, steadied over frames (see `stabilize`).
final class FaceParser {
    static let shared = FaceParser()
    static let side = 512                       // model input
    static let grid = 64                        // logits grid
    static let cropScale: CGFloat = 2.0         // crop side, of the face box (hair fits)
    static let faceClasses: Set<Int> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    static let skinClasses: Set<Int> = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]   // face only: no ears, hair, hat
    static let keep: Float = 0.45               // share of the previous frame in the average
    static let enter: Float = 0.6               // a cell joins the mask above this...
    static let leave: Float = 0.4               // ...and leaves it below this

    static var lastMs = 0.0                    // this frame's prediction, for the frame log
    private let model: MLModel?
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var input: CVPixelBuffer?
    // Previous frame, in crop cells (the crop follows the head, so a cell
    // stays on about the same spot of the head between frames).
    private var lastCrop = CGRect.null
    private var average = [Float]()
    private var inside = [Bool]()

    private init() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = Bundle.main.url(forResource: "FaceParsing", withExtension: "mlmodelc")
            .flatMap { try? MLModel(contentsOf: $0, configuration: config) }
        CVPixelBufferCreate(
            kCFAllocatorDefault, Self.side, Self.side, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &input)
    }

    var isAvailable: Bool {
        model != nil && input != nil
    }

    // Mask over `image` (white with alpha = face probability), or nil.
    // With a face box the network sees a square crop around the face
    // (as in its training set, CelebAMask-HQ): ~2x more grid cells on the
    // face than the whole frame, so a finer, steadier contour.
    func faceMask(_ image: CIImage, around face: CGRect = .null, classes: Set<Int> = faceClasses) -> CIImage? {
        guard let model, let input else {
            return nil
        }
        let start = CACurrentMediaTime()
        defer { Self.lastMs = (CACurrentMediaTime() - start) * 1000 }
        let extent = image.extent
        let crop: CGRect
        if face.isNull || face.width < 8 {
            let side = max(extent.width, extent.height)
            crop = CGRect(x: extent.minX, y: extent.minY, width: side, height: side)
        } else {
            // Centre raised by 10% of the face height: room for the hair above.
            let side = max(face.width, face.height) * Self.cropScale
            crop = CGRect(x: face.midX - side / 2, y: face.midY + face.height * 0.1 - side / 2,
                          width: side, height: side)
        }
        let scale = CGFloat(Self.side) / crop.width
        // Edge pixels repeated past the frame border, so the crop never sees black.
        let fitted = image.clampedToExtent()
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        context.render(fitted, to: input, bounds: CGRect(x: 0, y: 0, width: Self.side, height: Self.side),
                       colorSpace: CGColorSpaceCreateDeviceRGB())
        guard let features = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]),
              let output = try? model.prediction(from: features),
              let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return nil
        }
        let probability = stabilize(Self.faceProbability(logits, classes: classes), crop: crop)
        guard let small = Self.grayImage(probability, side: Self.grid) else {
            return nil
        }
        // 64 -> crop size: smoothed over ~half a cell first (no grid steps),
        // then a steeper edge around p = 0.5.
        let up = crop.width / CGFloat(Self.grid)
        return small.clampedToExtent()
            .applyingGaussianBlur(sigma: 0.6)
            .cropped(to: CGRect(x: 0, y: 0, width: Self.grid, height: Self.grid))
            .transformed(by: CGAffineTransform(scaleX: up, y: up))
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 3, y: 0, z: 0, w: 0),
                "inputBiasVector": CIVector(x: 1, y: 1, z: 1, w: -1),
            ])
            .applyingFilter("CIColorClamp")
            .transformed(by: CGAffineTransform(translationX: crop.minX, y: crop.minY))
            .cropped(to: extent)
    }

    // Against the edge jumping: the probability is averaged with the
    // previous frame, and a cell switches in or out of the mask only past
    // a margin (hysteresis: in above 0.6, out below 0.4, else as before).
    // Reset when the crop jumps (new face, big move or size change).
    private func stabilize(_ bytes: [UInt8], crop: CGRect) -> [UInt8] {
        let count = bytes.count
        let fresh = bytes.map { Float($0) / 255 }
        let steady = !lastCrop.isNull && average.count == count
            && abs(crop.width / lastCrop.width - 1) < 0.15
            && hypot(crop.midX - lastCrop.midX, crop.midY - lastCrop.midY) < crop.width * 0.1
        lastCrop = crop
        if !steady {
            average = fresh
            inside = fresh.map { $0 >= 0.5 }
        } else {
            for i in 0..<count {
                average[i] = average[i] * Self.keep + fresh[i] * (1 - Self.keep)
            }
        }
        var out = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            let p = average[i]
            if p > Self.enter {
                inside[i] = true
            } else if p < Self.leave {
                inside[i] = false
            }
            // Keep the soft value, but on the side of 0.5 that the state says.
            let shown = inside[i] ? max(p, 0.55) : min(p, 0.45)
            out[i] = UInt8(min(255, max(0, shown * 255)))
        }
        return out
    }

    // Softmax over the 19 classes per cell, summed over the face classes;
    // row 0 = top of the frame.
    static func faceProbability(_ logits: MLMultiArray, classes faceClasses: Set<Int>) -> [UInt8] {
        let classes = 19
        let cells = grid * grid
        var values = [Float](repeating: 0, count: classes * cells)
        // Direct buffer reads (NSNumber subscripts would cost milliseconds).
        let strides = logits.strides.map { $0.intValue }
        func gather<T: MLShapedArrayScalar>(_ type: T.Type, _ convert: (T) -> Float) {
            logits.withUnsafeBufferPointer(ofType: type) { buffer in
                for c in 0..<classes {
                    for y in 0..<grid {
                        for x in 0..<grid {
                            values[c * cells + y * grid + x] =
                                convert(buffer[c * strides[1] + y * strides[2] + x * strides[3]])
                        }
                    }
                }
            }
        }
        if logits.dataType == .float16 {
            gather(Float16.self) { Float($0) }
        } else {
            gather(Float.self) { $0 }
        }
        var out = [UInt8](repeating: 0, count: cells)
        for i in 0..<cells {
            var top: Float = -.greatestFiniteMagnitude
            for c in 0..<classes {
                top = max(top, values[c * cells + i])
            }
            var total: Float = 0
            var face: Float = 0
            for c in 0..<classes {
                let e = exp(values[c * cells + i] - top)
                total += e
                if faceClasses.contains(c) {
                    face += e
                }
            }
            out[i] = UInt8(min(255, max(0, face / total * 255)))
        }
        return out
    }

    // 8-bit gray bytes (row 0 = top) -> upright CIImage.
    static func grayImage(_ bytes: [UInt8], side: Int) -> CIImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                  width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: side,
                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            return nil
        }
        return CIImage(cgImage: cgImage)
    }
}
