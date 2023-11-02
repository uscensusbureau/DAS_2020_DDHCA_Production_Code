# ctools

A collection of python modules that are of general use.

## Installation

ctools is built and published as a python package.  It's hosted by DAS in
Nexus at https://repo.rm.census.gov/repository/DAS_Python/.

To install the latest version on DAS systems:

```
pip install --upgrade ctools
```

To install on non-DAS systems:

```
pip install --upgrade --extra-index-url http://repo.rm.census.gov/repository/DAS_Python/ ctools
```

### Versioning

ctools versioning uses a 4 digit semantic versioning convention:
```
<major>.<minor>.<patch>.<build>
```
A change to the major version number indicates a significant change in functionality that is not backwards compatible with previous versions.

A change to the minor version number indicates a backwards compatible change to functionality.

A change in the patch version indicates a minor bug fix or documentation change.

A change in the build version indicates a packaging related fix that does not change functionality.

## Development

### Dependency Management
Dependencies for ctools are defined in the requirements.txt file.  As part of the packaging process, the setup.cfg file will be generated from the setup.cf.template file combined with the requirements.txt file.

Only edit the dependencies in requirements.txt, never in setup.cfg.template.

Ensure that all packages defined in requirements.txt work with both python3.7 and 3.8.

### Version Updates
The version of the package is managed in __init__.py.

Update the MAJOR, MINOR and PATCH versions as appropriate and commit the change.

Note:  Whenever a higher level value is updated, reset the lower to 0. If you increase the MAJOR version, set MINOR, PATCH and BUILD to 0.  

### Building Locally

First, ensure you are in a virtual environment to isolate any dependencies/changes:

```
python3 -m venv .venv && .venv/bin/activate
```

Next, run make to build and install the package

```
make install
```

### Jenkins Builds

Builds that pass unit tests are automatically packaged and published to Nexus by Jenkins.

The main branch build publishes "official" packages to https://repo.rm.census.gov/repository/DAS_Python/.

Non-main branch builds will publish a test package to https://repo.rm.census.gov/repository/DAS-Pypi-Test/.


## Future Ideas
* Explore bottle websockets with https://pypi.org/project/bottle-websocket/
