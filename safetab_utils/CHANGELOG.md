# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 0.5.3 - 2023-05-30
### Changed
 - Removed references to Software Documentation that is internal for the Census.
 - Minor documentation updates.

## 0.5.2 - 2023-05-26
### Changed
 - Switched sample iteration codes to match Census iteration code standards.

## 0.5.1 - 2023-05-24
### Changed
 - Minor changes to documentation, types, and error messages.

## 0.5.0 - 2023-05-09
### Changed
- Updated some variable names for readability.
- Re-worded and expanded some docstrings and comments.

## 0.4.1 - 2023-04-20
### Changed
- Input files now required to be non-empty, but output files can be empty.

## 0.4.0 - 2023-04-17
### Added
- Add a check for duplicate population groups in the pop-group-totals input to the input validation.
- Updated to tmlt.common 0.7.3. 

### Changed
- Removed dependency on safetab-p and safetab-h code. Changed signature of `output_validation.validate_output`.
- Removed shim for old-style cef reader.
- Improved error message when cef reader import fails.

## 0.3.1 - 2022-01-17
### Changed
- Threshold validation error message is more informative.

## 0.3.0 - 2022-12-21
### Changed
- Updated common dependency for a newer version of numpy and pyarrow.

## 0.2.1 - 2022-12-08
### Changed
- Changed reader interface to work with either old or new cef reader.

## 0.2.0 - 2022-12-02
### Changed
- Changed reader interface to accept `safetab-p` or `safetab-h` as an additional required input parameter.
- Split reader interface into two separate objects, one for each program.

## 0.1.1 - 2022-09-09
### Fixed
- Updated tmlt.core and tmlt.analytics to fix a bug where queries failed to evaluate. 

## 0.1.0 - 2022-07-27
### Added
- New input validation check that race and ethnicity iteration code lists are disjoint.
- New project to house some shared utilities of the SafeTab products
- Update `TABBLKST` domain to be passed state_filter in input validation of GRF-C and records files
- Update list of valid FIPS codes for US run in `state_filter_us` check.
- Log errors instead of warning and fail input validation in case of race codes after a 'Null' and iteration hierarchy height checks.
- Add DP program output validation logic.
