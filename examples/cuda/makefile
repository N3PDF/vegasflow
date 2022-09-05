target_lib=integrand.so

TF_CFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2> /dev/null`
TF_LFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null`

CXX=g++
CXFLAGS=-std=c++17 -shared -fPIC -O2
KERNEL_DEF=-D KERNEL_CUDA=1
NCCFLAGS=-std=c++17 $(KERNEL_DEF) -x cu -Xcompiler -fPIC --disable-warnings

# Check whether there's nvcc
ifeq (,$(shell which nvcc 2>/dev/null))
else
	NCC:=nvcc
	NCCLIB:=$(subst bin/nvcc,lib64, $(shell which nvcc))
	CXFLAGS+=$(KERNEL_DEF) -L$(NCCLIB) -lcudart
	kernel_comp=integrand.cu.o
endif

.PHONY: run clean

run: $(target_lib)
	@python cuda_example.py

%.cu.o: %.cu.cpp
	@echo "[$(NCC)] Integrating cuda kernel..."
	@$(NCC) $(NCCFLAGS) -c -o $@ $< $(TF_CFLAGS)

%.so: %.cpp $(kernel_comp)
	@echo "[$(CXX)] Integrating operator..."
	@$(CXX) $(CXFLAGS) $(KERNEL) -o $@ $^ $(TF_CFLAGS) $(TF_LFLAGS)

clean:
	rm -f $(target_lib) $(kernel_comp)
