import CoreImage
import CoreML
import CoreVideo
import QuartzCore

// Per-frame face mask from a face-parsing network (YOLO26-nano, 19 classes,
// CelebAMask-HQ; Models/FaceParsing.mlpackage, compiled into the app by CI).
// Unlike the ARKit mesh, the mask comes from the very frame it is applied
// to, so it does not lag behind fast head turns. Face = skin, nose, glasses,
// eyes, brows, mouth, lips (no ears, hair, neck). Soft edge from the summed
// class probabilities.
final class FaceParser {
    static let shared = FaceParser()
    static let side = 512                       // model input
    static let grid = 64                        // logits grid
    static let faceClasses: Set<Int> = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

    static var lastMs = 0.0                    // this frame's prediction, for the frame log
    private let model: MLModel?
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var input: CVPixelBuffer?

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
    func faceMask(_ image: CIImage) -> CIImage? {
        guard let model, let input else {
            return nil
        }
        let start = CACurrentMediaTime()
        defer { Self.lastMs = (CACurrentMediaTime() - start) * 1000 }
        let extent = image.extent
        // The whole square frame scaled to 512 (no crop: faces near the edge keep logits).
        let scale = CGFloat(Self.side) / max(extent.width, extent.height)
        let fitted = image
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        context.render(fitted, to: input, bounds: CGRect(x: 0, y: 0, width: Self.side, height: Self.side),
                       colorSpace: CGColorSpaceCreateDeviceRGB())
        guard let features = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]),
              let output = try? model.prediction(from: features),
              let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return nil
        }
        let probability = Self.faceProbability(logits)
        guard let small = Self.grayImage(probability, side: Self.grid) else {
            return nil
        }
        // 64 -> frame size with smooth sampling, then a steeper edge around p = 0.5.
        let up = CGFloat(Self.side) / CGFloat(Self.grid) / scale
        return small
            .transformed(by: CGAffineTransform(scaleX: up, y: up))
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 3, y: 0, z: 0, w: 0),
                "inputBiasVector": CIVector(x: 1, y: 1, z: 1, w: -1),
            ])
            .applyingFilter("CIColorClamp")
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
    }

    // Softmax over the 19 classes per cell, summed over the face classes;
    // row 0 = top of the frame.
    static func faceProbability(_ logits: MLMultiArray) -> [UInt8] {
        let classes = 19
        let cells = grid * grid
        var values = [Float](repeating: 0, count: classes * cells)
        // Direct buffer reads (NSNumber subscripts would cost milliseconds).
        let strides = logits.strides.map { $0.intValue }
        func gather<T>(_ type: T.Type, _ convert: (T) -> Float) {
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
