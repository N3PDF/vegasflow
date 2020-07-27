//#include "cuda_kernel.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "integrand.h"

/* 
 * In this example we follow the TF guide for operation creation
 * https://www.tensorflow.org/guide/create_op
 * to create an integrand as a custom operators.
 *
 * To first approximation, these operators are function that take
 * a tensor and return a tensor.
 */

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

// CPU
template <typename T>
struct IntegrandOpFunctor<CPUDevice, T> {
    void operator()(const CPUDevice &d, const T *input, T *output, const int nevents, const int dims) {
        for (int i = 0; i < nevents; i++) {
            output[i] = 0.0;
            for(int j = 0; j < dims; j++) {
                output[i] += input[i,j];
            }
        }
    }
};


/* The input and output type must be coherent with the types used in tensorflow
 * at this moment we are using float64 as default for vegasflow.
 *
 * The output shape is set to be (input_shape[0], ), i.e., number of events
 */
//REGISTER_OP("IntegrandOp")
//    .Input("xarr: double")
//    .Output("ret: double")
//    .SetShapeFn([](shape_inference::InferenceContext* c) {
//        c -> set_output(0, c -> MakeShape( { c -> Dim(c -> input(0), 0) } ) );
//        return Status::OK();
//    });

template<typename Device, typename T>
class IntegrandOp: public OpKernel {
    public:
        explicit IntegrandOp(OpKernelConstruction* context): OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab input tensor, which is expected to be of shape (nevents, ndim)
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.tensor<T, 2>().data();
            auto input_shape = input_tensor.shape();

            // Create an output tensor of shape (nevents,)
            Tensor* output_tensor = nullptr;
            TensorShape output_shape;
            const int N = input_shape.dim_size(0);
            const int dims = input_shape.dim_size(1);
            output_shape.AddDim(N);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            auto output_flat = output_tensor->flat<T>().data();

            // Perform the actual computation
            IntegrandOpFunctor<Device, T>()(
                    context->eigen_device<Device>(), input, output_flat, N, dims
            );
        }
};

// Register the CPU version of the kernel
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("IntegrandOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), IntegrandOp<CPUDevice, T>);
REGISTER_CPU(double);

// Register the GPU version
#ifdef KERNEL_CUDA
#define REGISTER_GPU(T) \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(Name("IntegrandOp").Device(DEVICE_GPU).TypeConstraint<T>("T"),IntegrandOp<GPUDevice, T>);
REGISTER_GPU(double);
#endif
