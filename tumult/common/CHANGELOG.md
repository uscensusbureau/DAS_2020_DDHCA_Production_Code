# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 0.8.1 - 2023-05-26
### Changed
- Removed references to Software Documentation that is internal for the Census.

## 0.8.0 - 2023-05-10
### Changed
- Removed all references to ektelo, including renaming `Schema.from_ektelo_config`.
- Re-worded and expanded some docstrings and comments.
- Updated some variable names for readability.

## 0.7.5 - 2023-05-01
### Changed
- Remove references to statistical testing module from README.

## 0.7.4 - 2023-04-20
### Changed
- Made empty dataframe validation in `validate_spark_df` optional, default off, for backwards compatibility.
- Removed Statistical Testing Module.

## 0.7.3 - 2023-04-14
### Changed
- Changed logging messages during validation to say passed/failed "schema validation" instead of just passed/failed "validation"
- Added a validation check to validate_spark_df that requires data to exist in the dataframe. 

### Removed
- `ImplicitMatrix` and related code moved to a separate module, as it was not used across many Tumult products.

## 0.7.2 - 2023-03-15
### Fixed
- `Unrestricted` attributes' `validate` method now requires full regex matches.

### Changed
- Removed the tmlt.core dependency by creating a module to hold the PySparkTest class.

## 0.7.1 - 2023-02-16
### Changed
- Updated the smart_open library.

## 0.7.0 - 2022-12-21
### Changed
- Updated code to work with newer versions of numpy and pyarrow.

## 0.6.1 - 2022-07-27
### Changed
- Cast int/float values to str in `Unrestricted` attribute to fix AttributeError

## 0.6.0 - 2022-05-05
### Added
- Added test_requirements.txt file.
- Added schema.py, configuration.py, and validation.py

### Documentation
- Enabled Sphinx linkcheck and nitpicky mode
- Enabled building documentation for previous releases of the package.
- Updated documentation configuration and made miscellaneous changes to some docstrings.
- Future documentation will include any exceptions defined in this library.

### Changed
- Moved examples into `examples/` directory, from `common_examples`.

## 0.5.0 - 2021-09-10
### Added
- Test utils for geometric and discrete gaussian distributions.

### Changed
- Relocated resources folder inside package.
- Updated format and tutorials of software documentation.

## 0.4.0 - 2021-05-25
### Changed
- Namespaced packages as tmlt.

## 0.3.1 - 2021-04-27
### Fixed
- Updates to spark_test_harness to fix error causing DTYPE exceptions in tests that compare empty DataFrames and to set certain spark properties.


## 0.3.0 - 2021-03-31
### Added
- Pylint argument linting.

### Changed
- Helper functions used by PHSafe and SafeTab moved to common.
- Ding et al style statistical testing framework to use Spark.
- `write_log_file` and `get_logger_stream` added to `io_helpers` module.

## 0.2.0 - 2020-12-14
### Added
- Ding et al style statistical testing framework.
- Modules - `ImplicitMatrix`, `error`, `error_report` and  `io_helpers`.
- Seed added to `error_report` to make runs reproducible.
- Multiple input columns supported in `error_report`.
- Changelog.

### Changed
- READMEs updated.
- `marshallable` updated to handle tuples and lists separately.
- Reorganized tests, categorized into fast/slow tests, and increased coverage.
- Modules updated as per `mypy` linter for static type checking and `pylint` docstring arg linting.
- `example_scripts` directory renamed to `common_examples`.

### Deprecated
- Methods in `ImplicitMatrix.matrix` - Marginal.binary, Marginal.included_axes, and Marginal.excluded_axes.

## 0.1.2 - 2020-07-20
### Changed
- Boilerplate language

## 0.1.1 - 2020-07-14
### Changed
- Boilerplate language

## 0.1.0 - 2020-07-09
### Removed
- Modules - `ImplicitMatrix`, `error`, and `error_report`

## 0.1.0-alpha - 2020-06-12
### Added
- Modules - `ImplicitMatrix`, `error`, `error_report`, and `marshallable`.
