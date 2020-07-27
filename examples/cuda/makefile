target_cpu=integrand.so
cpu_src=integrand.cpp

target_gpu=integrand.cu.o
gpu_src=integrand.cu.cpp

TF_CFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2> /dev/null`
TF_LFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null`

CXX=g++
NCC=nvcc
CXFLAGS=-std=c++11 -shared -fPIC -O2
NCCFLAGS=-std=c++11 -D KERNEL_CUDA=1 -x cu -Xcompiler -fPIC
NLFLAGS=-L/opt/cuda/lib64 -lcudart

.PHONY: run_cpu run_gpu clean

run_cpu: $(target_cpu)
	@python cuda_example.py

run_gpu: $(target_gpu) $(target_cpu)
	@python cuda_example.py

$(target_cpu): $(cpu_src)
	@echo "Compiling CPU kernel"
	@$(CXX) $(CXFLAGS) $< -o $@ -fPIC $(TF_CFLAGS) $(TF_LFLAGS)

$(target_gpu): $(gpu_src)
	@echo "Compiling GPU kernel"
	@$(NCC) $(NCCFLAGS) -c -o $@ $< $(TF_CFLAGS)
	@$(CXX) $(CXFLAGS) -o $(target_cpu) $(cpu_src) $(target_gpu) $(TF_CFLAGS) $(NLFLAGS) $(TF_LFLAGS)

clean:
	rm -f $(target_gpu) $(target_gpu) integrand.so
