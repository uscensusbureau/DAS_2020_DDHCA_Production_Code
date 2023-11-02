# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 5.0.0
### Changed
- Updated to tmlt.analytics 0.5.3 to add support for pyspark 3.3.0 and 3.3.1, which are the default versions in EMR 6.8.0 and 6.10.0.
- `repo_zip.sh` no longer deletes pre-existing `__init__.py` files.
- Updated to tmlt.safetab_utils 0.5.3 and tmlt.common 0.8.1.
- Updated documentation instructions.
- Updated sample iteration codes to match Census iteration code standards.

## 5.0.0-beta
### Changed
- Removed the suggestion to use extraLibraryPath options on EMR.
- Updated license to Apache 2.0.
- Adjusted test constants to improve readability.
- Added public EMR install instructions.
- Adjusted example scripts for readability.
- Re-organized markdown documentation.
- Revised and expanded docstrings and comments.
- Renamed variables for clarity.

## 4.1.1 - 2022-12-08
### Fixed
- Fixed `examples/repo_zip.sh` to work with the production cef reader.

### Changed
- Modified safetab-p to accept either old or new reader interface. Note that the old reader interface is deprecated.

## 4.1.0 - 2022-12-02
### Changed
- Minor style changes to address Galois feedback.
- When a race code is mapped to multiple iteration codes in the same level, update error message to be more descriptive and include a list of offending codes.
- Added the ability to suppress rows with small counts.
- Changed reader interface to accept `safetab-p` as an additional required input parameter.
- Removed code "99999" from the output for PLACE.
- Update tmlt.core and switched to a wheel install process. 
- Updated tmtl.analytics.

## 4.0.1 - 2022-09-09
### Fixed
- Updated tmlt.core and tmlt.analytics to fix a bug where queries failed to evaluate. 

## 4.0.0 - 2022-07-27
### Added
- New system test for race and ethnicity disjointness input validation.
- Added Tract and Place support to PR runs of safetab-p.
- Add `--validate-private-output` flag to private `execute` mode in CLI.

### Changed
- Updated uses of `QueryBuilder.groupby_domains` to `QueryBuilder.groupby` operations using `KeySet`s.
- Renamed to `safetab_p` to exclusively house SafeTab-P product - SafeTab-P command-line, validation, DP and non-DP algorithms.

### Removed
- Removed unused options `random_seed` and `hardware_rng_mode` from config.
- Shared SafeTab code moved to `safetab_utils` and SafeTab-H code moved to `safetab_h` project.

## 3.1.0 - 2022-05-05
### Added
- Script to convert short form iterations to long form iterations.

### Changed
- Updated default toy dataset configs with max_race_codes=8 and allow_negative_count=true.
- Updated public join to new `QueryBuilder.join_public` interface.
- Tabulations updated to use single query evaluate of the Safetables `Session` interface rather than the deprecated multi-query variant.
- Updated uses of `Session.from_spark` to new interface that does not directly use `QueryExprCompiler`.
- Updated uses of `PrivacyBudget` to the new interface.
- Updated uses of `QueryBuilder.groupby` to the newly-named `QueryBuilder.groupby_domains`.
- Removed session-level noise param usage and specified the mechanism used in each query.
- Updated dependencies to use the renamed `tmlt.analytics` package.
- Moved examples and benchmarks into `examples/` and `benchmark/` directories, respectively, from `safetab_examples` and `safetab_benchmark`.
- Updated Tumult library name (was: "SafeTables", now: "Analytics")
- Added ability to assign a threshold to each population group level for SafeTab-P using config.
- Added support for additional geos. SafeTab now supports following geos: USA, State, County, Tract, Place and AIANNH.
- T1/T2 post-processing: total population for every characteristic iteration is output in T1 and the sex marginal for every characteristic iteration eligible for T2 statistics is output in T2.
- Add repartition to write output as single part csv than multiple part files per t1/t2/t3/t4. If US and PR are run together, two separate CSVs are created.
- Add more checks when validating config `state_filter_us` key.
- Run config validation prior to private algorithm execution of SafeTab-P and SafeTab-H.

### Fixed
- HHRACE input domains for SafeTab-H household-records input validation updated to allow values 01-63 and error on Null.

### Documentation
- Enabled building documentation for previous releases of the package.
- Future documentation will include any exceptions defined in this library.

## 3.0.0 - 2021-09-10
### Added
- Added `privacy_defn` flag for switching between PureDP and Rho zCDP to the config

### Changed
- Namespaced packages as tmlt and relocated resources folder inside package.
- Epsilons in config changed to privacy_budgets.
- Tabulations updated to run on Safetables `Session` interface.
- Refactoring of nonprivate algorithm to be directly on spark.

### Removed
- Removed SafeTab H statistical privacy test.

## 3.0.0-alpha - 2021-04-27
## Added
- Added SafeTab csv reader and support for CEF reader.

## Changed
- Updates SafeTab-H to use `ProtectedSparkDataFrame`.
- Moved some utils to common package.

## Fixed
- Fixed bug in SafeTab-H that disregarded flatmap sensitivity.

## 2.0.0 - 2020-12-14
## Added
- safetab-p algorithm updated to new specifications.
- US vs Puerto Rico implementation to safetab-p algorithm.

### Changed
- SafeTab updated in accordance with Ektelo changes to: schema management, localizing of configs, and categorical attribute
- Accuracy report and Ding test updated.
- Reorganized tests, categorized into fast/slow tests, and increased coverage.
- Modules updated as per `mypy` linter for static type checking and `pylint` docstring arg linting.
- `example_scripts` directory renamed to `safetab_examples`

### Fixed
- `unpersist` cache call at the end of safetab-p

## 1.1.1 - 2020-09-22
### Fixed
- `spark_local_properties.conf`, `example_scripts` and tests updated to run on AWS EMR.
- READMEs updated with AWS EMR specific instructions.

## 1.1.0 - 2020-09-14
## Added
- Fix for race condition that used to rewrite the configs in `/tmp/safetab`.

### Changed
- Spark used in `input_validation`.
- Logging statements.

## 1.1.0-alpha - 2020-08-31
### Added
- SafeTab-P algorithm using spark in `plan_p_spark`.
- SafeTab-H command-line.
- SafeTab-H DP and non-DP algorithms.
- SafeTab-H Ding et al style statistical test.
- Changelog.

### Changed
- `input_validation` updated to fail when input files, except GRF-C.txt, have unexpected columns.
- `input_validation` updated to check column ordering and includes SafeTab-H validation.
- SafeTab-P command-line updated to include `--use-spark` option.
- Input validation call added to ground truth algorithms to avoid file not found errors.
- `exclude_states_from_usrun`, `run_us`, `run_pr` options added to `config.json`.
- SafeTab `requirements.txt` updated to include pyarrow, pyspark and smart_open.

## 1.0.0 - 2020-08-05
### Added
- SafeTab-P Ding et al style statistical test.

### Changed
- SafeTab-P command-line create log file directory if it doesn't exist.

## 1.0.0-alpha - 2020-07-23
### Added
- SafeTab-P algorithms.
- Compute population thresholds and workloads from SafeTEx formulas.

### Changed
- SafeTab-P command-line interface includes algorithm execution command.

## 0.1.2 - 2020-07-20
### Changed
- Boilerplate language

## 0.1.1 - 2020-07-14
### Changed
- Boilerplate language

## 0.1.0 - 2020-07-09
### Added
- SafeTab-P validation utility that can run on all the full inputs and make sure they satisfy the specification.
