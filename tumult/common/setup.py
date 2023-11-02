# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.common']

package_data = \
{'': ['*']}

install_requires = \
['boto3>=1.16.0,<2.0.0',
 'numpy>=1.21.0,<1.21.6',
 'pandas>=1.2.0,<2.0.0',
 'pyarrow>=6.0.0,<7.0.0',
 'pyspark[sql]>=3.0.0,<3.1.0',
 'scipy>=1.4.1,<1.6.1',
 'smart_open>=5.2.0,<6.0.0']

setup_kwargs = {
    'name': 'tmlt-common',
    'version': '0.8.1',
    'description': 'Common utility functions used by Tumult projects',
    'long_description': '# Common Utility\n\nThis module primarily contains common utility functions used by different Tumult projects.\n\n<placeholder: add notice if required>\n\n## Overview\n\nThe utility functions include:\n* Methods to serialize/deserialize objects into json format (marshallable).\n* Expected error computations.\n* A tool for creating error reports.\n\nSee [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.\n\n## Testing\n\nTo run the tests, install the required dependencies from the `test_requirements.txt`\n\n```\npip install -r test_requirements.txt\n```\n\n*All tests (including Doctest):*\n\n```bash\nnosetests tmlt/common --with-doctest\n```\n\nSee `examples` for examples of features of `common`.\n',
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.4,<3.8.0',
}


setup(**setup_kwargs)
