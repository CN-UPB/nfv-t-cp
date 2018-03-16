#!/bin/bash

#
# temporary helper to kick-off a multi core run
#
# ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9

nfvppsim -c configs/experiment_tc_paper_synthetic_l2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l3.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l3.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l5.yaml "$@" &

nfvppsim -c configs/experiment_tc_paper_synthetic_d2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d4.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d5.yaml "$@" &
