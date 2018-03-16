#!/bin/bash

#
# temporary helper to kick-off a multi core run
#

nfvppsim -c configs/experiment_tc_paper_synthetic_l2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l3.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l3.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_l5.yaml "$@" &

nfvppsim -c configs/experiment_tc_paper_synthetic_d2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d2.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d4.yaml "$@" &
nfvppsim -c configs/experiment_tc_paper_synthetic_d5.yaml "$@" &
