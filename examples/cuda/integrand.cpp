#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

/* 
 * In this example we follow the TF guide for operation creation
 * https://www.tensorflow.org/guide/create_op
 * to create an integrand as a custom operators.
 *
 * To first approximation, these operators are function that take
 * a tensor and return a tensor.
 */

using namespace tensorflow;

REGISTER_OP("IntegrandOp")
.Input("xarr: double")
.Output("ret: double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c -> set_output(0, c -> MakeShape( { c -> Dim(c -> input(0), 0) } ) );
        return Status::OK();
        });

class IntegrandOp: public OpKernel {
    public:
        explicit IntegrandOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            // the input tenjsor is expected to have a (nevents, ndim) shape
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.tensor<double, 2>();
            auto input_shape = input_tensor.shape();

            // Create an output tensor
            // the expected shape is (nevents,)
            Tensor* output_tensor = NULL;
            TensorShape output_shape;
            const int N = input_shape.dim_size(0);
            output_shape.AddDim(N);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
            auto output_flat = output_tensor->flat<double>();

            // Sum in the dimensional axis
            for (int i = 0; i < N; i++) {
                output_flat(i) = 0.0;
                for(int j = 0; j < input_shape.dim_size(1); j++) {
                    output_flat(i) += input(i,j);
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("IntegrandOp").Device(DEVICE_CPU), IntegrandOp);
