#!/bin/bash


LOG_DIR="./logs/"

TENSORS=(
  # lbnl 
  # nips 
  # uber 
  # chicago
  vast 
  # darpa 
  enron 
  # nell-2 
  # fb-m 
  flickr-4d 
  delicious-4d 
  # nell-1 
  # amazon
)

get_tns_name() {
  declare -A tns_files=(
    ["lbnl"]="lbnl"
    ["nips"]="nips"
    ["uber"]="uber"
    ["chicago"]="chicago-crime-comm"
    ["vast"]="vast-2015-mc1-5d"
    ["enron"]="enron"
    ["flickr-4d"]="flickr-4d" 
    ["delicious-4d"]="deicious-4d"
  )
  echo "${tns_files[${1}]}.tns"
}

get_streaming_mode() {
  declare -A streaming_mode=(
    ["lbnl"]="4"
    ["nips"]="3"
    ["uber"]="0"
    ["chicago"]="0"
    ["vast"]="0"
    ["enron"]="0"
    ["flickr-4d"]="3" 
    ["delicious-4d"]="3" 
  )
  echo "${streaming_mode[${1}]}"
}

VARIATIONS=(
  spcpstream 
  spcpstream_alto 
  cpstream_alto 
  cpstream
)

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

make 
rm logs/*

for ((i=0; i < ${#TENSORS[@]}; ++i))
do
  for ((v=0; v < ${#VARIATIONS[@]}; ++v))
  do
    # Generate command line
    PARTITION="fat"
    TENSOR=${TENSORS[i]}
    JOBNAME="$TENSOR_${VARIATIONS[v]}"
    OUTPUT="$LOG_DIR${TENSORS[i]}_${VARIATIONS[v]}.out"
    ERROR="$LOG_DIR${TENSORS[i]}_${VARIATIONS[v]}.err"
    TNS_FILE=$(get_tns_name $TENSOR)
    STREAMING_MODE=$(get_streaming_mode $TENSOR)
    echo -e "output logged in: $OUTPUT\n"
    echo -e "error loggin in : $ERROR\n"
    # CMD="./cpd128 --rank 16 -m 300 -a ${STREAMING_MODE[i]} -e 1e-5 -l ${VARIATIONS[v]} -i ~/hpctensor/${TENSORS[i]}.tns > ${TENSORS[i]}.${VARIATIONS[v]}.log.txt"
    CMD="./cpd128 --rank 16 -m 300 -a $STREAMING_MODE -e 1e-5 -l ${VARIATIONS[v]} -i ~/hpctensor/$TNS_FILE"
    echo -e "$CMD\n"
    # Replace __VARIABLES__ in template.srun and pass through bash
    sed "s@__PARTITION__@$PARTITION@g; s@__JOBNAME__@$JOBNAME@g; s@__OUTPUT__@$OUTPUT@g; s@__ERROR__@$ERROR@g; s@__CMD__@$CMD@g" ./scripts/template.srun | sbatch
  done
done

# STREAMING_MODE=$(get_streaming_mode $dataset)

# echo "Running ${TNS_FILE}-streaming mode ${STREAMING_MODE}"

# sed "s/__PARTITION__/fat/g; s/__JOBNAME__/uber/g; s/__CMD__/echo $TNS_FILE ssup/g" ./template.srun | bash
