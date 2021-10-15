#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

make

rm *.log.txt

#TENSORS=(lbnl nips uber chicago vast darpa enron nell-2 fb-m flickr-4d delicious-4d nell-1 amazon)
#MODES = (...)

TENSORS=(uber chicago-crime-comm delicious-4d flickr-4d)

for ((i=0; i < ${#TENSORS[@]}; ++i))
do
    echo "Running tensor ${TENSORS[i]}..." >> test.log.txt
    ./cpd128 --rank 16 -m 100 -a 0 -l cpstream_alto -x 44 -i ~/hpctensor/${TENSORS[i]}.tns >> alto.${TENSORS[i]}.log.txt #2>&1
    ./cpd128 --rank 16 -m 100 -a 0 -l cpstream -x 44 -i ~/hpctensor/${TENSORS[i]}.tns >> non_alto.${TENSORS[i]}.log.txt #2>&1
done