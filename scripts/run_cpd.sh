#!/bin/bash

############################# HELPER FUNCTIONS ################################
#_______________________________________________
echoerr() {
    echo -e "$@" 1>&2
}
#_______________________________________________

#_______________________________________________
# Retuns: the maximum number of modes (zero-based) for a given tensor.
# Usage: get_modes TENSOR
get_modes () {
    declare -A target_modes=(
        ["amazon"]="2" ["chicago"]="3" ["darpa"]="2" ["delicious-4d"]="3" ["enron"]="3"
        ["fb-m"]="2" ["flickr-4d"]="3" ["lbnl"]="4" ["nell-1"]="2" ["nell-2"]="2"
        ["nips"]="3" ["patents"]="2" ["reddit"]="2" ["uber"]="3" ["vast"]="4"
    )
    echo "${target_modes[${1}]}"
}
#_______________________________________________

#_______________________________________________
# Returns: the appropriate binary name, i.e., "cpd128" if the given tensor requires a
#          128-bit ALTO mask and "cpd64" otherwise.
# Usage: get_binary TENSOR
get_binary () {
    declare -A bitmask_size=( ["amazon"]="128" ["chicago"]="64" ["darpa"]="64" ["delicious-4d"]="128" ["enron"]="64" ["fb-m"]="64" ["flickr-     4d"]="128" ["lbnl"]="128" ["nell-1"]="128" ["nell-2"]="64" ["nips"]="64" ["patents"]="64" ["reddit"]="64" ["uber"]="64" ["vast"]="64")
    echo "cpd${bitmask_size[${1}]}"
}
#_______________________________________________

#_______________________________________________
# Usage: write_log STRING
write_log () {
    INDENT=`printf '=%.0s' {1..20}`
    echo -n "$(date '+%Y-%m-%d_%H:%M:%S'):   " >> $LOGF
    echo -e $1 >> $LOGF
}
#_______________________________________________
#_______________________________________________
# Usage: write_logf FILEPATH
write_logf () {
    INDENT=`printf ' %.0s' {1..23}`
    echo -n "$(date '+%Y-%m-%d_%H:%M:%S'):   " >> $LOGF
    head -n 1 $1 >> $LOGF
    cat $1 | awk -v indent="$INDENT" 'NR > 1 {print indent $0}' >> $LOGF
}
#_______________________________________________

#_______________________________________________
check_for_tensors() {
    if [ -z "$TENSOR_DIR"]; then
        echoerr -e "Variable \$TENSOR_DIR must be set and point to the directory containing the tensors.\nExit."
        exit 1
    fi
}
#_______________________________________________

#_______________________________________________
download_tensors() {
    declare -A tensor_urls=(
        ["amazon"]="-o amazon.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz"
        ["chicago"]="-o chicago.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz"
        ["darpa"]="-o darpa.tar.gz https://datalab.snu.ac.kr/data/haten2/1998DARPA.tar.gz"
        ["delicious-4d"]="-o dleicious-4d.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-4d.tns.gz"
        ["enron"]="-o enron.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz"
        ["fb-m"]="-o fb-m.tar.gz https://datalab.snu.ac.kr/data/haten2/freebase_music.tar.gz"
        ["flickr-4d"]="-o flickr-4d.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-4d.tns.gz"
        ["lbnl"]="-o lbnl.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/lbnl-network/lbnl-network.tns.gz"
        ["nell-1"]="-o nell-1.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz"
        ["nell-2"]="-o nell-2.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz"
        ["nips"]="-o nips.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz"
        ["patents"]="-o patents.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/patents/patents.tns.gz"
        ["reddit"]="-o reddit.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/reddit-2015/reddit-2015.tns.gz"
        ["uber"]="-o uber.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz"
        ["vast"]="-o vast.tns.gz https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-5d.tns.gz"
    )
    mkdir -p $TENSOR_DIR
    cd $TENSOR_DIR
    for tns in $TENSORS; do
        curl --retry 5 --retry-delay 1 -C ${tensor_urls[${tns}]}
    done
}
#_______________________________________________
#_______________________________________________
extract_tensors() {
    cd $TENSOR_DIR
    shopt -s nullglob
    for gzfile in *.gz; do
        echo "Decompress $gzfile"
        gzip -d -f $gzfile
    done
    for tarfile in *tar; do
        echo "Decompress $tarfile"
        tar -xf $tarfile
        rm -f $tarfile
    done
    # special treatment for DARPA and FB-M
    HATEN_TENSORS="1998DARPA freebase_music"
    declare -A  haten_dict=( ["1998DARPA"]="darpa" ["freebase_music"]="fb-m" )
    for htn_tns in `echo $HATEN_TENSORS`; do
        if [[ -d "$htn_tns" ]]; then
            mv ${htn_tns}/${htn_tns}.tensor ./${haten_dict[${htn_tns}]}.tns
            rm -rf $htn_tns
        fi
    done
}
#_______________________________________________

measure_tensors() {
    # set OMP environment if not set
    if [[ -z "$OMP_NUM_THREADS" ]]; then
        export OMP_NUM_THREADS="$(grep -c ^processor /proc/cpuinfo)"
    fi
    if [[ -z "$OMP_PLACES" ]]; then
        socks="$(lscpu | grep "Socket(s)" | awk '{print $NF}')"
        threads_per_core="$(lscpu | grep "Thread(s) per core" | awk '{print $NF}')"
        cores_per_sock="$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')"
        export OMP_PLACES="\"{0:$((threads_per_core * cores_per_sock))}:$((threads_per_core * cores_per_sock)):${socks}\""
    fi
    if [[ -z "$OMP_STACKSIZE" ]]; then
        export OMP_STACKSIZE=4000M
    fi
    if [[ -z "$OMP_WAIT_POLICY" ]]; then
        export OMP_WAIT_POLICY=active
    fi

    cd $BASE_DIR
    TMPLOG=".tmp.${DATETAG}.log"
    write_log "START================================================================"
    write_log "Run $TENSORS, store results in $OUTF."
    write_log "ENV Vars: OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PLACES=$OMP_PLACES OMP_STACKSIZE=$OMP_STACKSIZE OMP_WAIT_POLICY=$OMP_WAIT_POLICY"

    echo "tensor,mode,mttkrp_time,format_gen,da_mem" >> $OUTF
    for tensor in $TENSORS; do
        binary=$( get_binary $tensor )

        if [[ "$1" == "mttkrp" ]]; then
            # do MTTKRP
            echo -n "Run ${tensor} mode "
            max_mode=$( get_modes $tensor )
            for mode in `seq 0 $max_mode`; do
                write_log "Run ${tensor} mode ${mode}"
                echo -n "${mode}..."
#                $omp_n_threads $omp_pl $omp_stacks $omp_wait_pol ./${binary} --rank 16 -m 100 -p -t $mode -i ${TENSOR_DIR}/${tensor}.tns > $TMPLOG 2>&1
                ./${binary} --rank 16 -m 100 -p -t $mode -i ${TENSOR_DIR}/${tensor}.tns > $TMPLOG 2>&1
                write_logf $TMPLOG
                mttkrp_t=`cat $TMPLOG | grep "ALTO runtime" | awk '{print $3}'`
                format_gen_t=`cat $TMPLOG | grep "ALTO creation time" | awk '{print $4}'`
                da_mem_t=`cat $TMPLOG | grep "ALTO da_mem creation time" | awk '{print $5}'`
                echo "${tensor},${mode},${mttkrp_t},${format_gen_t},${da_mem_t}">> $OUTF
            done
            echo "done"
        else
            # do CPD
            echo "Run ${tensor}..."
            echo "tensor,time[s],mttkrp[s],pseudoinv[s],memcpy[s],normalize[s],update[s],fit[s],iterations,s/it,fit-val,fit-delta" >> $OUTF
#            $omp_n_threads $omp_pl $omp_stacks $omp_wait_pol ./${binary} --rank 16 -m 100 -i ${TENSOR_DIR}/${tensor}.tns > $TMPLOG 2>&1
            ./${binary} --rank 16 -m 100 -i ${TENSOR_DIR}/${tensor}.tns > $TMPLOG 2>&1
            write_logf $TMPLOG
            TOTLINE=`cat $TMPLOG | tail -n 3 | head -n 1 | sed -r "s/   /,/g"`
            NUM_ITS=`cat $TMPLOG | tail -n 6 | head -n 1 | awk '{print $2+1}'`
            FIT=`cat $TMPLOG | tail -n 6 | head -n 1 | awk '{print $4}'`
            FIT_DELTA=`cat $TMPLOG | tail -n 6 | head -n 1 | awk '{print $6}'`
            TOT_TIME=`cat $TMPLOG | tail -n 3 | head -n 1 | awk '{print $1}'`
            TIME_PER_IT=`printf '%.6f\n' "$(echo "$TOT_TIME / $NUM_ITS" | bc -l)"`
            echo "${tensor},${TOTLINE},${NUM_ITS},${TIME_PER_IT},${FIT},${FIT_DELTA}" >> $OUTF
        fi
    done
    rm $TMPLOG

    write_log "FINISH==============================================================="
}

#_______________________________________________
print_help() {
    echo    "Usage: ./run_cpd.sh [-h] [-t TENSORS] -d | -m | -c"
    echo -e "\nPositional arguments:"
    echo    "  -d, --download       Download and extract set of tensor to ${BASE_DIR}/tensors/"
    echo    "  -m, --mttkrp         "
    echo    "  -c, --cpd            "
    echo -e "\nOptional arguments:"
    echo    "  -t TENSOR,           Apply process defined by positional arguments only on a list of tensors."
    echo    "  --tensor TENSOR      The tensors must be findable under this name in \"\$TENSOR_DIR\".By default, "
    echo    "                       the tensor list is \"lbnl nips uber chicago vast darpa enron nell-2 fb-m flickr-4d"
    echo    "                       delicious-4d nell-1 amazon patents reddit\" and TENSOR_DIR relates to ${BASE_DIR}/tensors/."
    echo    "  --tensor-dir PATH    Set \"\$TENSOR_DIR\" to PATH"
}
#_______________________________________________

###############################################################################

BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
DATETAG="$(date '+%Y-%m-%d_%H-%M-%S')"
EXPORTF="./.export_vars.sh"
TENSORS="lbnl nips uber chicago vast darpa enron nell-2 fb-m flickr-4d delicious-4d nell-1 amazon patents reddit"

# Check parameters
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
        print_help
        exit 0
        ;;
    -d|--download)
        DOWNLOAD_TENSORS="1"
        shift
        ;;
    -m|--mttkrp)
        DO_MTTKRP="1"
        shift
        ;;
    -c|--cpd)
        DO_CPD="1"
        shift
        ;;
    -t|--tensor)
        TENSORS="$2"
        shift
        shift
        ;;
    --tensor-dir)
        TENSOR_DIR="$2"
        shift
        shift
        ;;
    *)  # unknown option
        shift
        ;;
esac
done

if [[ -z "$DO_MTTKRP" &&  -z "$DO_CPD" && -z "$DOWNLOAD_TENSORS" ]]; then
    echoerr "No positional argument given."
    print_help
    exit 0
fi

if [[ "$DO_MTTKRP" == "1" ]]; then
    # do mttkrp
    echo "MTTKRP for $TENSORS"
    LOGF="${BASE_DIR}/res_mttkrp_${DATETAG}.log"
    OUTF="${BASE_DIR}/res_mttkrp_${DATETAG}.csv"
    measure_tensors mttkrp
fi

if [[ "$DO_CPD" == "1" ]]; then
    # do cpd als
    echo "CPD ALS for $TENSORS"
    LOGF="${BASE_DIR}/res_cpd_${DATETAG}.log"
    OUTF="${BASE_DIR}/res_cpd_${DATETAG}.csv"
    measure_tensors
fi

if [[ -z "$TENSOR_DIR" ]]; then
    TENSOR_DIR="${BASE_DIR}/tensors"
fi

if [[ "$DOWNLOAD_TENSORS" == "1" ]]; then
    # download tensors from FROSTT and HaTen2
    echo "Tensors will be downloaded into ${BASE_DIR}/tensors"
    echo "ALTO does not own any tensors and will download tensors from the FROSTT and HaTen2 repository."
    echo "This requires approx. 240 GB. If you want to download only specific tensors, please use the --tensor option."
    echo "Currently selected tensor(s): ${TENSORS}"
    read -p "Continue? (Y/N) "
    case "$REPLY" in
        y|Y)
            echo "Download ${TENSORS}..."
            download_tensors
            echo "Extract tensors..."
            extract_tensors
            ;;
        n|N)
            exit 0
            ;;
        *)
            echo -e "Please type Y or N.\nExit"
            exit 1
            ;;
    esac
fi
