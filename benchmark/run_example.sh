#!/bin/bash

# example=drellyan_lo_tf.py
example=lepage_tf.py
nevents=$(( 4*10**8 ))
limit=$(( 10**8 ))


cpu_events=$(( ${nevents}/10 ))
cpu_limit=$(( ${limit}/10 ))

# Run CPU times:
echo "### CPU benchmark ###"
export CUDA_VISIBLE_DEVICES=""
for i in {0..35}
do
    echo  "> > Running for $(( i+1 )) cores"
    taskset -c 0-${i} python ${example} -n ${nevents} -l ${limit} -q 2>/dev/null
    echo ""
done

echo "### GPU benchmark ###"
echo " CPU limited to 0-4 "
echo " > > Running on Titan V"
export CUDA_VISIBLE_DEVICES=0
taskset -c 0-3 python ${example} -n ${nevents} -l ${limit} -q 2>/dev/null

echo " > > Running on RTX 2080 Ti"
export CUDA_VISIBLE_DEVICES=1
taskset -c 0-3 python ${example} -n ${nevents} -l ${limit} -q 2>/dev/null

echo " > > Running on Titan V and RTX 2080 Ti"
export CUDA_VISIBLE_DEVICES=0,1
taskset -c 0-7 python ${example} -n ${nevents} -l ${limit} -q 2>/dev/null
