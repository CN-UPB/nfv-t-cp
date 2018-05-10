[![Build Status](https://travis-ci.org/CN-UPB/nfv-t-cp.svg?branch=master)](https://travis-ci.org/CN-UPB/nfv-t-cp)

# nfv-t-cp: NFV Time-Constrained Profiler

NFV time-constrained performance profiling (T-CP) framework.

This framework can be used to build NFV profiling systems that profile single VNFs or complex SFC under a given time constrained. The framework can be either connected to a real-world profiling platform or it can be fed with existing profiling results to perform trace-based T-CP runs independently of a connected profiling platform.

## Contact


Manuel Peuster<br>
Paderborn University<br>
manuel (dot) peuster (at) upb (dot) de<br>
Twitter: @ManuelPeuster<br>
GitHub: @mpeuster<br>

## Installation

```bash
python3 setup.py install
```

## Run (single job)

```bash
nfvtcp -c example_experiment.yaml
```

## Run (parallel jobs)

```bash
./nfvtcp_parallel.sh -c example_experiment.yaml -J 16
```
The argument `-J` specifies the number of parallel jobs to use.

### Combine results of parallel jobs

If the data frames of multiple parallel jobs should be combined do:

```bash
cd out/
python combine_results.py -i "experiment_*.pkl" -o combined_result.pkl
```
NOTE: The quotes for the file pattern are important!

## Build-in plotting

```bash
nfvtcp -c example_experiment.yaml --plot out/result.pkl
```

## Tests

To run `nfv-t-cp`'s unit tests do:

```bash
pytest
pytest -v -s --log-level DEBUG # be more verbose
pytest -v -s -k "midpoint" # run tests that match keyword
```

### Examples

```bash
# run
./nfvtcp_parallel.sh -J 16 -c configs/experiment_tc_paper_synthetic_all.yaml -r 100
./nfvtcp_parallel.sh -J 16 -c configs/experiment_tc_paper_nfvsdn17_measurements.yaml -r 100

# process results
python combine_results.py -i "experiment_tc_paper_synthetic_all.job*" -o 2018-04-XX-experiment_tc_paper_synthetic_all.compressed.combined.pkl
python combine_results.py -i "experiment_tc_paper_nfvsdn17_measurements.job*" -o 2018-04-XX-experiment_tc_paper_nfvsdn17_measurements.combined.compressed.pkl
```

## Contributor(s)

* Manuel Peuster (manuel (at) peuster (dot) de)
* ...

## License

Apache 2.0
