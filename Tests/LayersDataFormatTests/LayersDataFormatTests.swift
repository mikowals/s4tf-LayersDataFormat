import XCTest
import TensorFlow
@testable import LayersDataFormat

final class LayersDataFormatTests: XCTestCase {
    func testAveragePooledDF() {
        let input = Tensor<Float>(randomNormal: [12, 16, 16, 3])
        let targetNHWC = avgPool2D(input,
                               filterSize: (1, 2, 2, 1),
                               strides: (1, 2, 2, 1),
                               padding: .same)
        XCTAssertEqual(input.averagePooledDF(kernelSize: (1, 2, 2, 1),
                                             strides: (1, 2, 2, 1),
                                             padding: .same,
                                             dataFormat: .nhwc),
                       targetNHWC)
        /* TODO. Uncomment .nchw test that will only be run on GPU.
        XCTAssertEqual(input.transposed(withPermutations: [0, 3, 1, 2])
                            .averagePooledDF(kernelSize: (1, 2, 2, 1),
                                             strides: (1, 2, 2, 1),
                                             padding: .same,
                                             dataFormat: .nchw),
                       targetNHWC.transposed(withPermutations: [0, 3, 1, 2]))
        */
    }
    
    func testConvolved2DDF() {
        let filter = Tensor<Float>(randomNormal: [3, 3, 3, 16])
        let inputNHWC = Tensor<Float>(randomNormal: [12, 16, 16, 3])
        let targetNHWC = conv2D(inputNHWC,
                            filter: filter,
                            strides: (1, 1, 1, 1),
                            padding: .same)
        
        for device in [Device.defaultTFEager, Device.defaultXLA] {
            for dataFormat in [_Raw.DataFormat.nhwc] {
                let input, target: Tensor<Float>
                switch dataFormat {
                case .nchw:
                    input = inputNHWC.transposed(permutation: [0, 3, 1, 2])
                    target = targetNHWC.transposed(permutation: [0, 3, 1, 2])
                case .nhwc:
                    input = inputNHWC
                    target = targetNHWC
                }
                XCTAssertEqual(input.convolved2DDF(withFilter: filter,
                                                   strides: (1,1,1,1),
                                                   padding: .same,
                                                   dataFormat: dataFormat),
                               target,
                               "Conv2D fails.  DataFormat: \(dataFormat) Device: \(device)")
                
            }
        }
    }

    static var allTests = [
        ("testConvolved2DDF", testConvolved2DDF),
    ]
}
