# SafeTab-P Tests

SafeTab-P and supporting libraries provide a range of tests to ensure that the software is working correctly.

Our tests are divided into unit and system tests. Unit tests verify that the implementation of a class, and its associated methods, match the behavior specified in its documentation. System tests are designed to determine whether the assembled system meets its specifications.

We also divide tests into fast and slow tests. Fast tests complete relatively quickly, and can be run often, while slow tests are longer-running and less frequently exercised. While unit tests tend to be fast and system tests tend to be slow, there are some slow unit tests and fast system tests.

Tests for SafeTab-P, Common, and SafeTab-Utils are run using nose. Tests for Analytics are run using pytest. Core is provided as a binary wheel, and thus does not have runnable tests in this release.

All tests are run on a single machine. Runtimes mentioned in this document were measured on an `r4.16xlarge` machine.

## Running all tests

Execute the following to run the tests:

*Fast Tests:*

```bash
DIR=<path of cloned repository>
python3 -m nose $DIR/common -a '!slow'
python3 -m pytest $DIR/analytics -m 'not slow'
python3 -m nose $DIR/safetab_p -a '!slow'
python3 -m nose $DIR/safetab_utils -a '!slow'
(Total runtime estimate: 20 minutes)
```

*Slow Tests:*

```bash
DIR=<path of cloned repository>
python3 -m nose $DIR/common -a 'slow'
python3 -m pytest $DIR/analytics -m 'slow'
python3 -m nose $DIR/safetab_p -a 'slow'
python3 -m nose $DIR/safetab_utils -a 'slow'
(Total runtime estimate: 140 minutes)
```

## SafeTab-P's tests

### Unit Tests:

SafeTab-P unit tests test the individual components of the algorithm like characteristic iteration flat maps, region preprocessing, and accuracy report components.

```
python3 -m nose safetab_p/test/unit
(Runtime estimate: 45 seconds)
```

### System Tests:
#### **Input Validation**:

   * Tests that the input to SafeTab-P matches input specification.
   * Tests that SafeTab-P raises appropriate exceptions when invalid arguments are supplied and that SafeTab-P runs the full algorithm for US and/or Puerto Rico depending on input supplied.

```
python3 -m nose safetab_p/test/system/test_input_validation.py
(Runtime estimate: 3 minutes)
```

 The below spark-submit commands demonstrates running SafeTab-P command line program in input `validate` mode on toy dataset and a csv reader. To validate with the CEF reader, see instructions in the [README](./README.md).

*Run from the directory `safetab_p/tmlt/safetab_p`.*

```bash
spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        ./safetab-p.py validate resources/toy_dataset/input_dir_<puredp or zcdp> \
         resources/toy_dataset
(Runtime estimate: 35 seconds)
```

#### **Output Validation**:

* Tests that the output to SafeTab-P conforms to the output specification and varies appropriately with changes in the input. Also tests that flags for non-negativity and excluding states work as expected.

The below spark-submit commands demonstrates running SafeTab-P command line program to produce private tabulations followed by output validation on toy dataset and a csv reader.

*Run from the directory `safetab_p/tmlt/safetab_p`.*

```bash
spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        ./safetab-p.py execute resources/toy_dataset/input_dir_<puredp or zcdp>  \
        resources/toy_dataset  \
        example_output/safetab_p --validate-private-output
(Runtime estimate: 4 minutes)
```

#### **Correctness Tests (also test of consistency)**:

   * A test checks that when all values of privacy budget goes to infinity, the output converges to the correct (noise-free/ground truth) answer.

   * A test ensures that the SafeTab-P algorithm outputs the correct population groups (characteristic iteration and geographic entity) with the correct aggregate counts when non-zero privacy budget is allocated to the iteration/geo level.

   * A test ensures that the SafeTab-P algorithm determines the correct statistic level and computes the correct aggregate counts when the noise addition is turned off.

```
python3 -m nose safetab_p/test/system/test_correctness.py
(Runtime estimate: 2 hours and 10 minutes)
```

#### **Accuracy Test**:

   * Compares noisy results to a ground-truth (non-private) calculations of the answer.

   * Calculates the observed error between the noisy and ground-truth results.

Run [`safetab_p/examples/multi_run_error_report_p.py`](examples/multi_run_error_report_p.py) to compare the results from SafeTab-P algorithm execution against the ground truth (non-private) answers across multiple trials and privacy budgets. This example script runs the SafeTab-P program on non-sensitive data present in input path (`safetab_p/tmlt/safetab_p/resources/toy_dataset`) for 5 trials, epsilons specified in [`safetab_p/tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json`](tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json) for each combination of geography level and iteration level. It produces ground truth tabulations `t1/*.csv` and `t2/*.csv` in directory (`safetab_p/example_output/multi_run_error_report_p/ground_truth`). Private tabulations `t1/*.csv` and `t2/*.csv` for each run can be found in the directory (`safetab_p/example_output/multi_run_error_report_p/single_runs`). The aggregated error report `multi_run_error_report.csv` is saved to output directory (`safetab_p/example_output/multi_run_error_report_p/full_error_report`).


```bash
safetab_p/examples/multi_run_error_report_p.py
(Runtime estimate: 15 minutes)
```

Note: Multi-run error report uses the ground truth counts. It violates differential privacy, and should not be created using sensitive data. Its purpose is to test SafeTab-P on non-sensitive or synthetic datasets to help tune the algorithms and to predict the performance on the private data.

   * Tests that running the full accuracy report creates the appropriate output directories for SafeTab-P algorithm.

```
python3 -m nose safetab_p/test/system/test_accuracy_report.py
(Runtime estimate: 25 minutes)
```
