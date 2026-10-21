import ARKit
import UIKit
import CoreImage
import Metal
import SceneKit

// 3D face overlays on the TrueDepth face mesh (ARKit). ARKit owns the
// camera here: each ARFrame gives the camera image and the face anchor.
// The overlay is rendered by SceneKit into a transparent texture and
// composited over the camera image; the result goes to RoundOutput.
final class FaceSource: NSObject, FrameSource, ARSessionDelegate {
    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    private let output: RoundOutput
    private let session = ARSession()
    private let queue = DispatchQueue(label: "roundlab.face")
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let context: CIContext
    private let renderer: SCNRenderer
    private let scene = SCNScene()
    private let cameraNode = SCNNode()
    private var style: FaceStyle
    private var faceNodes: [StyledNode] = []
    private var overlay: MTLTexture?
    private var depth: MTLTexture?
    private var lastTimestamp: TimeInterval?

    init?(output: RoundOutput, style: FaceStyle) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }
        self.output = output
        self.style = style
        self.device = device
        self.commandQueue = commandQueue
        context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
        renderer = SCNRenderer(device: device, options: nil)
        super.init()

        let camera = SCNCamera()
        camera.zNear = 0.001
        camera.zFar = 10
        cameraNode.camera = camera
        scene.rootNode.addChildNode(cameraNode)
        scene.background.contents = nil
        renderer.scene = scene
        renderer.pointOfView = cameraNode
        renderer.autoenablesDefaultLighting = false
        session.delegate = self
        session.delegateQueue = queue
    }

    func setStyle(_ next: FaceStyle) {
        queue.async { [self] in
            style = next
            faceNodes.forEach { $0.node.removeFromParentNode() }
            faceNodes = []
        }
    }

    func start() {
        let configuration = ARFaceTrackingConfiguration()
        configuration.maximumNumberOfTrackedFaces = 1
        configuration.isLightEstimationEnabled = false
        let formats = ARFaceTrackingConfiguration.supportedVideoFormats.filter { $0.framesPerSecond == 30 }
        if let format = formats.min(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            configuration.videoFormat = format
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        session.pause()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let started = CACurrentMediaTime()
        let side = output.side
        let viewport = CGSize(width: side, height: side)
        let intervalMs = lastTimestamp.map { (frame.timestamp - $0) * 1000 } ?? 0
        lastTimestamp = frame.timestamp

        let face = frame.anchors
            .compactMap { $0 as? ARFaceAnchor }
            .first { $0.isTracked }

        // Camera image in the same portrait square the projection uses.
        let background = Self.cameraImage(frame, viewport: viewport)
        var composed: CIImage
        if let face {
            cameraNode.camera?.projectionTransform = SCNMatrix4(frame.camera.projectionMatrix(
                for: .portrait,
                viewportSize: viewport,
                zNear: 0.001,
                zFar: 10))
            cameraNode.simdTransform = frame.camera.viewMatrix(for: .portrait).inverse
            updateNodes(face)
            guard let layer = renderOverlay(time: frame.timestamp, side: side) else {
                composed = SignalEffect.lostScreen(
                    extent: CGRect(origin: .zero, size: viewport),
                    noise: SignalEffect.staticNoise(extent: CGRect(origin: .zero, size: viewport), time: frame.timestamp, brightness: 0.22),
                    time: frame.timestamp)
                deliver(composed, frame: frame, started: started, intervalMs: intervalMs, faces: 0, lost: true)
                return
            }
            composed = style.finish(layer.composited(over: style.backdrop(background)), time: frame.timestamp)
        } else {
            faceNodes.forEach { $0.node.isHidden = true }
            let extent = CGRect(origin: .zero, size: viewport)
            composed = SignalEffect.lostScreen(
                extent: extent,
                noise: SignalEffect.staticNoise(extent: extent, time: frame.timestamp, brightness: 0.22),
                time: frame.timestamp)
        }
        deliver(composed, frame: frame, started: started, intervalMs: intervalMs, faces: face == nil ? 0 : 1, lost: face == nil)
    }

    private func deliver(_ image: CIImage, frame: ARFrame, started: CFTimeInterval, intervalMs: Double, faces: Int, lost: Bool) {
        let side = output.side
        guard let target = output.makeBuffer() else {
            return
        }
        context.render(
            image.cropped(to: CGRect(x: 0, y: 0, width: side, height: side)),
            to: target,
            bounds: CGRect(x: 0, y: 0, width: side, height: side),
            colorSpace: CGColorSpaceCreateDeviceRGB())
        let time = CMTime(seconds: frame.timestamp, preferredTimescale: 1_000_000_000)
        output.deliver(
            target,
            at: time,
            duration: CMTime(value: 1, timescale: 30),
            stats: RoundOutput.Stats(
                intervalMs: intervalMs,
                trackMs: 0,
                renderMs: (CACurrentMediaTime() - started) * 1000,
                faces: faces,
                held: false,
                lost: lost))
    }

    private func updateNodes(_ face: ARFaceAnchor) {
        if faceNodes.isEmpty {
            faceNodes = style.makeNodes(device: device)
            faceNodes.forEach { scene.rootNode.addChildNode($0.node) }
        }
        for styled in faceNodes {
            (styled.node.geometry as? ARSCNFaceGeometry)?.update(from: face.geometry)
            let scale = simd_float4x4(diagonal: SIMD4<Float>(styled.scale, styled.scale, styled.scale, 1))
            styled.node.simdTransform = face.transform * scale
            styled.node.isHidden = false
        }
    }

    // SceneKit overlay with a transparent background.
    private func renderOverlay(time: TimeInterval, side: Int) -> CIImage? {
        if overlay == nil {
            let color = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: side, height: side, mipmapped: false)
            color.usage = [.renderTarget, .shaderRead]
            color.storageMode = .private
            overlay = device.makeTexture(descriptor: color)
            let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: side, height: side, mipmapped: false)
            depthDescriptor.usage = .renderTarget
            depthDescriptor.storageMode = .private
            depth = device.makeTexture(descriptor: depthDescriptor)
        }
        guard let overlay, let depth,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return nil
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = overlay
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .dontCare
        renderer.render(
            atTime: time,
            viewport: CGRect(x: 0, y: 0, width: side, height: side),
            commandBuffer: commandBuffer,
            passDescriptor: pass)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed,
              let image = CIImage(mtlTexture: overlay, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]) else {
            return nil
        }
        // Metal textures start at the top left, Core Image at the bottom left.
        return image.transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -CGFloat(side)))
    }

    // ARKit camera image mapped into the portrait viewport (same mapping
    // ARKit uses for projectionMatrix(for:viewportSize:)).
    static func cameraImage(_ frame: ARFrame, viewport: CGSize) -> CIImage {
        let pixels = frame.capturedImage
        let imageSize = CGSize(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels))
        let normalize = CGAffineTransform(scaleX: 1 / imageSize.width, y: 1 / imageSize.height)
        let flip = CGAffineTransform(scaleX: -1, y: -1).translatedBy(x: -1, y: -1)
        let display = frame.displayTransform(for: .portrait, viewportSize: viewport)
        let toViewport = CGAffineTransform(scaleX: viewport.width, y: viewport.height)
        return CIImage(cvPixelBuffer: pixels)
            .transformed(by: normalize.concatenating(flip).concatenating(display).concatenating(toViewport))
            .cropped(to: CGRect(origin: .zero, size: viewport))
    }
}

// A node that follows the face mesh, scaled around the head origin.
struct StyledNode {
    var node: SCNNode
    var scale: Float
}

// A 3D face look: SceneKit nodes on the face mesh plus 2D finishing.
protocol FaceStyle {
    var name: String { get }
    func makeNodes(device: MTLDevice) -> [StyledNode]
    func backdrop(_ camera: CIImage) -> CIImage
    func finish(_ image: CIImage, time: Double) -> CIImage
}

// "Ghost in the mesh": the face is a dark mask that follows the head and
// the expression; a thin glowing wireframe of the mesh lies on top.
struct GhostStyle: FaceStyle {
    let name = "ghost_mesh"
    var wire = UIColor(red: 0.92, green: 0.95, blue: 1.0, alpha: 0.85)

    func makeNodes(device: MTLDevice) -> [StyledNode] {
        guard let fill = ARSCNFaceGeometry(device: device, fillMesh: true),
              let lines = ARSCNFaceGeometry(device: device, fillMesh: false) else {
            return []
        }
        let dark = SCNMaterial()
        dark.diffuse.contents = UIColor(white: 0.03, alpha: 1)
        dark.lightingModel = .constant
        fill.firstMaterial = dark
        let fillNode = SCNNode(geometry: fill)
        fillNode.renderingOrder = 0

        let glow = SCNMaterial()
        glow.diffuse.contents = wire
        glow.lightingModel = .constant
        glow.fillMode = .lines
        glow.writesToDepthBuffer = false
        lines.firstMaterial = glow
        let lineNode = SCNNode(geometry: lines)
        lineNode.renderingOrder = 1
        // A little larger than the face so the edges stay covered.
        return [StyledNode(node: fillNode, scale: 1.06), StyledNode(node: lineNode, scale: 1.065)]
    }

    func backdrop(_ camera: CIImage) -> CIImage {
        camera
    }

    func finish(_ image: CIImage, time: Double) -> CIImage {
        let extent = image.extent
        return SignalEffect.scanlines(extent: extent).composited(over: image).cropped(to: extent)
    }
}

let allFaceStyles: [FaceStyle] = [GhostStyle()]
