#! /bin/bash
SLEEP=30

if [ -z ${BENCHMARK_NS} ]; then
    BENCHMARK_NS="default"
fi

if [ -z ${RUN_SET} ]; then
    RUN_SET="."
fi

mkdir -p status

workload_name() {
    BENCHMARK=$1
# uncomment the following lines to use benchmark full name
    # GOVERNOR=$([ -f "/sys/devices/system/cpu/cpu0/cpufreq" ] && cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor||echo unknown)
    # MAXFREQ=$([ -f "/sys/devices/system/cpu/cpu0/cpufreq" ] && cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq||echo unknown)
    # FIRST_VALUE=$(kubectl get benchmark $BENCHMARK -n${BENCHMARK_NS} -ojson|jq -r .spec.iterationSpec.iterations[0].values[0])
    # LAST_VALUE=$(kubectl get benchmark $BENCHMARK -n${BENCHMARK_NS} -ojson|jq -r .spec.iterationSpec.iterations[0].values[-1])
    # REPETITION=$(kubectl get benchmark $BENCHMARK -n${BENCHMARK_NS} -ojson|jq -r .spec.repetition)rep
    # WORKLOAD_NAME=${BENCHMARK}_${FIRST_VALUE}_to_${LAST_VALUE}_${REPETITION}_${GOVERNOR}_${MAXFREQ}
    # echo $WORKLOAD_NAME
# uncomment the following lines to use only benchmark name
    echo $BENCHMARK
}

save_benchmark() {
    BENCHMARK=$1
    WORKLOAD_NAME=$(workload_name $BENCHMARK)
    echo "saved as $WORKLOAD_NAME"
    kubectl get benchmark $BENCHMARK -n ${BENCHMARK_NS} -oyaml > status/${WORKLOAD_NAME}.yaml
}

deploy_benchmark() {
    BENCHMARK=$1
    kubectl create -f ${RUN_SET}/cpe_${BENCHMARK}.yaml
}

clean_benchmark() {
    BENCHMARK=$1
    kubectl delete -f ${RUN_SET}/cpe_${BENCHMARK}.yaml
}

expect_num() {
    BENCHMARK=$1
    kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -oyaml > tmp.yaml
    BENCHMARK_FILE=tmp.yaml
    num=$(cat ${BENCHMARK_FILE}|yq ".spec.repetition")
    if [ -z $num ]; then
        num=1
    fi

    for v in $(cat ${BENCHMARK_FILE}|yq eval ".spec.iterationSpec.iterations[].values | length")
    do
        ((num *= v))
    done
    rm tmp.yaml
    echo $num
}

wait_for_benchmark() {
    BENCHMARK=$1
    EXPECT_NUM=$(expect_num ${BENCHMARK})
    jobCompleted=$(kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -ojson|jq -r .status.jobCompleted)
    echo "Wait for ${EXPECT_NUM} ${BENCHMARK} jobs to be completed, sleep 1m"
    while [ "$jobCompleted" != "${EXPECT_NUM}/${EXPECT_NUM}" ] ; 
    do  
        sleep 60
        echo "Wait for ${BENCHMARK} to be completed... $jobCompleted, sleep 1m"
        jobCompleted=$(kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -ojson|jq -r .status.jobCompleted)
    done
    echo "Benchmark job completed"
}

configure_freq() {
    echo "configure CPU Frequency (wait for ${SLEEP}s)"
    FREQ=$1
    cpupower set -b 0
    cpupower frequency-set -d ${FREQ} 2>&1 > /dev/null
    cpupower frequency-set -u ${FREQ} 2>&1 > /dev/null
    sleep ${SLEEP}
}

run_benchmark() {
    BENCHMARK=$1
    # clean_benchmark ${BENCHMARK}
    # deploy_benchmark ${BENCHMARK}
    wait_for_benchmark ${BENCHMARK}
    save_benchmark ${BENCHMARK}
    clean_benchmark ${BENCHMARK}
}

run_trl_nx12() {
    BENCHMARK=coremark
    clean_benchmark ${BENCHMARK}
    for FREQ in "1200MHz" "1600MHz" "2000MHz" "2400MHz" "2800MHz" "3200MHz" "3600MHz"
    do
        configure_freq ${FREQ}
        run_benchmark ${BENCHMARK}
    done
}

run_all() {
    # BENCHMARK=coremark
    # run_benchmark ${BENCHMARK}

    # BENCHMARK=parsec
    # run_benchmark ${BENCHMARK}

    BENCHMARK=stressng
    run_benchmark ${BENCHMARK}
}


"$@"