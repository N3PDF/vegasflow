target_cpu=integrand.so
cpu_src=integrand.cpp

TF_CFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2> /dev/null`
TF_LFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null`

CXX=g++
CXFLAGS=-std=c++14 -shared -fPIC -O2

run_cpu: $(target_cpu)
	@python cuda_example.py

$(target_cpu): $(cpu_src)
	@echo "Compiling CPU kernel"
	@$(CXX) $(CXFLAGS) $< -o $@ -fPIC $(TF_CFLAGS) $(TF_LFLAGS)
