#!/bin/bash

module load intel/19.0.5
source /packages/intel/19.0.5/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux
make && ./cpd64 --rank 16 -m 100 -a 0 -i ~/hpctensor/uber.tns
make && ./cpd64 --rank 16 -m 100 -a 3 -i ~/hpctensor/delicious-4d.tns
make && ./cpd64 --rank 16 -m 100 -a 3 -i ~/hpctensor/flickr-4d.tns
#make && ./cpd64 --rank 16 -m 100 -i ~/hpctensor/uber.tns
