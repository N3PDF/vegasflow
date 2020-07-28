#ifndef KERNEL_INTEGRAND_
#define KERNEL_INTEGRAND_

namespace tensorflow {
    using Eigen::GpuDevice;

    template<typename Device, typename T>
    struct IntegrandOpFunctor {
        void operator()(const Device &d, const T *input, T *output, const int nevents, const int dims);
    };

#if KERNEL_CUDA
    template<typename T>
    struct IntegrandOpFunctor<Eigen::GpuDevice, T> {
        void operator()(const Eigen::GpuDevice &d, const T *input, T *output, const int nevents, const int dims);
    };
#endif

}

#endif
