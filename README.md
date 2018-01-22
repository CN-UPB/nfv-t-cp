# nfv-pp-sim

NFV performance profiling simulator. Used for research work on time-constraint NFV profiling (tc-profiling).

This repo is intended to be open sourced once the paper is submitted.

## Installation

* `python3 setup.py develop`

## Run

* `nfvppsim -c example_experiment.yaml`

## Plot

* `nfvppsim --plot out/2018-01-22_09-06_test.pkl`

## Test

* `py.test ...`

## Profiling

* `python -m cProfile -s cumtime run_profiling.py > profile.out`


## Contributor(s)

Manuel Peuster <manuel (at) peuster (dot) de

## License

Apache 2.0
