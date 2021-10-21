#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

make
# rm *.log.txt

#TENSORS=(lbnl nips uber chicago vast darpa enron nell-2 fb-m flickr-4d delicious-4d nell-1 amazon)
#MODES = (...)

#TENSORS=(uber chicago-crime-comm delicious-4d flickr-4d)
TENSORS=(uber)
STREAMING_MODE=(0)
#STREAMING_MODE=(0 0 3 3)

for ((i=0; i < ${#TENSORS[@]}; ++i))
do
    # Add log that writes down start time echo "" >> test.log.txt
#    ./cpd128 --rank 16 -m 100 -a ${STREAMING_MODE[i]} -l cpstream_alto -x 44 -i ~/hpctensor/${TENSORS[i]}.tns >> test.alto.${TENSORS[i]}.log.txt 2>&1
#    ./cpd128 --rank 16 -m 100 -a ${STREAMING_MODE[i]} -l cpstream -x 44 -i ~/hpctensor/${TENSORS[i]}.tns >> test.non_alto.${TENSORS[i]}.log.txt 2>&1
    ./cpd128 --rank 16 -m 100 -a ${STREAMING_MODE[i]} -l cpstream_alto -x 44 -i ~/hpctensor/${TENSORS[i]}.tns
#    ./cpd128 --rank 16 -m 100 -a ${STREAMING_MODE[i]} -l cpstream -x 44 -i ~/hpctensor/${TENSORS[i]}.tns >> test.non_alto.${TENSORS[i]}.log.txt 2>&1
done
