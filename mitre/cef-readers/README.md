# das-phsafe-cef-reader
Safetab CEF Reader


## Development
This package supports an editable install for easy development and debugging.

To install locally, run the following command.  It is recommended to do this inside a virtual environment to isolate 
any changes/dependencies.
Once this command is run, you can edit the code in src/phsafe_safetab_reader and it will be immediately available 
for use in python as if installed

[//]: # (via a package.)

```
pip3 install --force-reinstall -r requirements.txt
```

## Tests
Tests are included in the tests folder.  Run, from the root directory:
```
./tests/run_safetab_tests.sh
```
Note that the tests must be run on a system with Spark.

### Versioning

das-phsafe-cef-reader versioning uses a 4 digit semantic versioning convention:
```
<major>.<minor>.<patch>.<build>
```
A change to the major version number indicates a significant change in functionality that is not backwards compatible 
with previous versions.

A change to the minor version number indicates a backwards compatible change to functionality.

A change in the patch version indicates a minor bug fix or documentation change.

A change in the build version indicates a packaging related fix that does not change functionality.

#### Version Updates
The version of the package is managed in __init__.py.

Update the MAJOR, MINOR and PATCH versions as appropriate and commit the change.

Note:  Whenever a higher level value is updated, reset the lower to 0. If you increase the MAJOR version, set MINOR, 
PATCH and BUILD to 0.

## Package Information


This is used for the SafeTab and PHSafe CEF Readers, which convert the CEF microdata files into Spark Dataframes that 
SafeTab and PHSafe use as inputs.

This also contains code that adds variance information to the SafeTab output as a stopgap until SafeTab updates to 
include variance information, as well as code that stores the CEF data in a record layout format on s3.

SafeTab Documentation for the format of its inputs is [here](../tumult/safetab_p/SafeTab_P_Documentation.pdf)

### Important Files:

**safetab_cef_config_2010.ini** - Used for safetab P Cef Reader (safetab_cef_reader). It contains the locations of the 
2010 CEF microdata files on s3 which are the inputs for the CEF Reader

**safetab_cef_reader.py** - Reads the fixed-width CEF files in and uses the CEF Validator to turn them into dataframes. 
Does additional modifications to get dataframe to SafeTab's expected input.

**safetab_h_cef_reader.py** - Reads fixed-width CEF files and the outputs T1 files from 
Safetab-H, and uses the CEF Validator to turn them into dataframes. Does additional modifications to get dataframe 
to SafeTab's expected input.

**safetab_p_cef_reader.py** - Reads fixed-width CEF files and the outputs T1 files from
Safetab-P, and uses the CEF Validator to turn them into dataframes. Does additional modifications to get dataframe
to SafeTab's expected input. 

**cef_reader.py** - Reads the fixed-width CEF files in and uses the CEF Validator to turn them into dataframes. 
Does additional modifications to get dataframe to PHSafe's expected input.

**cef_validator_classes.py** A dependency for some of the python files, originally from the reader in das_decennial's 
programs/reader directory.
