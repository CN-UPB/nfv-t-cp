#!/bin/bash
#
# Copyright (c) 2018 Manuel Peuster
# ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Manuel Peuster, Paderborn University, manuel@peuster.de
#
# helper to kick-off a multi core run
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
    echo "nfvtcp -J $JOB_NO -j $i --log log/job$i.log $@"
    nfvtcp -J "$JOB_NO" -j "$i" --log log/job"$i".log "$@" &
done
echo "Started $JOB_NO simulation jobs."
