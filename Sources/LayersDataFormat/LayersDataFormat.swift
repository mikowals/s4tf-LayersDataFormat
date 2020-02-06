import TensorFlow

public extension Padding {
    @inlinable
    internal var raw: _Raw.Padding {
        switch self {
        case .same: return .same
        case .valid: return .valid
        }
    }

    @inlinable
    internal var raw2: _Raw.Padding2 {
        switch self {
        case .same: return .same
        case .valid: return .valid
        }
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// TensorFlow builtin conv2d gradient helper for the input.
    @inlinable
    internal func conv2DBackpropInputDF(
        shape: Tensor<Int32>,
        filter: Tensor,
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> Tensor {
        return _Raw.conv2DBackpropInput(
            inputSizes: shape,
            filter: filter,
            outBackprop: self,
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
            padding: padding.raw2,
            explicitPaddings: [],
            dataFormat: dataFormat)
    }

    /// TensorFlow builtin conv2d gradient helper for the filter.
    @inlinable
    internal func conv2DBackpropFilterDF(
        input: Tensor,
        filterSizes: Tensor<Int32>,
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> Tensor {
        return _Raw.conv2DBackpropFilter(
            input,
            filterSizes: filterSizes,
            outBackprop: self,
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
            padding: padding.raw2,
            explicitPaddings: [],
            dataFormat: dataFormat
        )
    }
    
    @inlinable
    @derivative(of: convolved2DDF, wrt: (self, filter))
    internal func _vjpConvolved2DDF(
        filter: Tensor,
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let value = convolved2DDF(withFilter: filter, strides: strides,
                                padding: padding, dataFormat: dataFormat)
        return (value, { v in
            return (
                v.conv2DBackpropInputDF(
                    shape: self.shapeTensor, filter: filter,
                    strides: strides, padding: padding, dataFormat: dataFormat
                ),
                v.conv2DBackpropFilterDF(
                    input: self, filterSizes: filter.shapeTensor,
                    strides: strides, padding: padding, dataFormat: dataFormat
                )
            )
        })
    }
    @inlinable
    @derivative(of: averagePooledDF, wrt: self)
    internal func _vjpAveragePooledDF(
        kernelSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        // TODO: Currently this is not higher order differentiable. Redefine in
        // closed form.
        let value = averagePooledDF(kernelSize: kernelSize, strides: strides,
                                  padding: padding, dataFormat: dataFormat)
        return (value, { v in
            return _Raw.avgPoolGrad(
                origInputShape: self.shapeTensor,
                grad: v,
                ksize: [Int32(kernelSize.0), Int32(kernelSize.1),
                        Int32(kernelSize.2), Int32(kernelSize.3)],
                strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
                padding: padding.raw,
                dataFormat: dataFormat
            )
        })
    }
}
public extension Tensor where Scalar: FloatingPoint {
    /// Computes a 2-D convolution using `self` as input, with the specified
    /// filter, strides, and padding.
    ///
    /// - Parameters:
    ///     - filter: The convolution filter.
    ///     - strides: The strides of the sliding filter for each dimension of the
    ///         input.
    ///     - padding: The padding for the operation.
    /// - Precondition: `self` must have rank 4.
    /// - Precondition: `filter` must have rank 4.
    @inlinable @inline(__always)
    func convolved2DDF(
        withFilter filter: Tensor,
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> Tensor {
        return _Raw.conv2D(
            self,
            filter: filter,
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
            padding: padding.raw2,
            explicitPaddings: [],
            dataFormat: dataFormat
        )
    }
    
    @inlinable @inline(__always)
    func averagePooledDF(
        kernelSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding,
        dataFormat: _Raw.DataFormat = .nhwc
    ) -> Tensor {
        return _Raw.avgPool(
            value: self,
            ksize: [Int32(kernelSize.0), Int32(kernelSize.1),
                    Int32(kernelSize.2), Int32(kernelSize.3)],
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
            padding: padding.raw,
            dataFormat: dataFormat)
    }
}
