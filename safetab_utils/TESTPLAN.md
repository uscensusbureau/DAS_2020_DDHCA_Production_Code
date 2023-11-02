# SafeTab-Utils Test Plan

All tests are run on a single machine. Runtimes mentioned in this document represent estimates on an `r4.16xlarge` machine.

### Unit Tests:

Unit tests verify that the implementation of a class, and its associated methods, match the behavior specified in its documentation.

SafeTab-Utils unit tests test the individual utilities like characteristic iteration flat maps, region preprocessing, config validation, input validation, CSV reader and other utility methods.

```
nosetests test/unit
(Runtime estimate: 15 seconds)
```

##### **Race codes and characteristic iterations**:

   * Test cases for the race/ethinicity code to characteristic iteration code mapping can be found in [`race_eth_codes_to_iterations_testcases.txt`](./tmlt/safetab_utils/resources/test/race_eth_codes_to_iterations_testcases.txt). These test cases are used by `TestRaceEthCodesToIterations` test class to test that race and ethnicity codes are correctly mapped to characteristic iterations.


```
nosetests test/unit/test_characteristic_iterations.py:TestRaceEthCodesToIterations
(Runtime estimate: less than 1 second)
```

##### **Config Validation**:

   * Tests that the input config to SafeTab matches input specification.

```
nosetests test/unit/test_config_validation.py
(Runtime estimate: 10 seconds)
```

### System Tests:

System tests are designed to determine whether the assembled system meets its specifications.
