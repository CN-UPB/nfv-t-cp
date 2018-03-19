#!/bin/bash

#
# temporary helper to kick-off a multi core run
#
# ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9

nfvppsim -c configs/experiment_tc_paper_synthetic_l2.yaml --log log/l2.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l3.yaml --log log/l3.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l4.yaml --log log/l4.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l5.yaml --log log/l5.log "$@" &

nfvppsim -c configs/experiment_tc_paper_synthetic_d2.yaml --log log/d2.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d3.yaml --log log/d3.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d4.yaml --log log/d4.log "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d5.yaml --log log/d5.log "$@" &
