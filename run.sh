#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux
# make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/uber.tns
# make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/chicago-crime-comm.tns
# make && ./cpd64 --rank 16 -m 100 -a 3 -i ~/hpctensor/flickr-4d.tns
# make && ./cpd128 --rank 16 -m 100 -a 3 -x 44 -i ~/hpctensor/sm_flickr.tns
# make && ./cpd128 --rank 16 -m 100 -a 3 -x 44 -i ~/hpctensor/flickr-4d.tns
# make && ./cpd64 --rank 16 -m 100 -i ~/hpctensor/sm_flickr.tns
# make && ./cpd128 --rank 16 -m 100 -a 3 -i ~/hpctensor/flickr-4d.tns

# make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/uber.tns
# make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/chicago-crime-comm.tns
# make && ./cpd64 --rank 16 -m 100 -a 3 -i ~/hpctensor/flickr-4d.tns
# make && ./cpd128 --rank 16 -m 100 -a 3 -l cpstream -x 44 -i ~/hpctensor/sm_flickr.tns
make && ./cpd64 --rank 16 -m 100 -a 0 -l cpstream -x 44 -i ~/hpctensor/uber.tns
# make && ./cpd64 --rank 16 -m 100 -a 3 -x 44 -i ~-l cpstream /hpctensor/sm_flickr.tns
# make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/uber.tns
# make && ./cpd128 --rank 16 -m 100 -a 3 -i ~/hpctensor/delicious-4d.tns
