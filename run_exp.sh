#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

make
# rm *.log.txt

#TENSORS=(lbnl nips uber chicago vast darpa enron nell-2 fb-m flickr-4d delicious-4d nell-1 amazon)
#MODES = (...)

TENSORS=(uber chicago-crime-comm nips delicious-4d flickr-4d)
STREAMING_MODE=(0 0 3 3 3)
VARIATIONS=(spcpstream spcpstream_alto cpstream_alto cpstream)
# VARIATIONS=(spcpstream_alto)

for ((i=0; i < ${#TENSORS[@]}; ++i))
do
  for ((v=0; v < ${#VARIATIONS[@]}; ++v))
  do
    ./cpd128 --rank 16 -m 300 -a ${STREAMING_MODE[i]} -e 1e-5 -l ${VARIATIONS[v]} -x 44 -i ~/hpctensor/${TENSORS[i]}.tns > ${TENSORS[i]}.${VARIATIONS[v]}.log.txt 2>&1 
  done
done
