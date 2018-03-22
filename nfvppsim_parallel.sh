#!/bin/bash
#
# temporary helper to kick-off a multi core run
#
# ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9

set -e

# parse argument (-J or --jobno) (other arguments are maintained)
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -J|--jobno)
    JOB_NO="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# kick-off simulations
for (( i=0; i<$JOB_NO; i++ ))
do
    echo "nfvppsim -J $JOB_NO -j $i --log log/job$i.log $@"
    nfvppsim -J "$JOB_NO" -j "$i" --log log/job"$i".log "$@" &
done
echo "Started $JOB_NO simulation jobs."
