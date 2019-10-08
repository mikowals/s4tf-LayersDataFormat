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
        let input = Tensor<Float>(randomNormal: [12, 16, 16, 3])
        let filter = Tensor<Float>(randomNormal: [3, 3, 3, 16])
        let targetNHWC = conv2D(input, filter: filter, strides: (1, 1, 1, 1), padding: .same)
        XCTAssertEqual(input.convolved2DDF(withFilter: filter,
                                           strides: (1,1,1,1),
                                           padding: .same,
                                           dataFormat: .nhwc),
                       targetNHWC)
        
        // TODO. Add .nchw test that will only be run on GPU.
    }

    static var allTests = [
        ("testConvolved2DDF", testConvolved2DDF),
    ]
}
