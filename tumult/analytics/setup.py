# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.analytics', 'tmlt.analytics._query_expr_compiler']

package_data = \
{'': ['*']}

install_requires = \
['pandas>=1.2.0,<2.0.0',
 'pyspark[sql]>=3.0.0,<3.3.1',
 'sympy>=1.8,<1.10',
 'tmlt.core>=0.6.0,<0.7.0',
 'typeguard>=2.12.1,<2.13.0',
 'typing-extensions>=3.10.0,<4.0.0']

setup_kwargs = {
    'name': 'tmlt-analytics',
    'version': '0.5.3',
    'description': "Tumult's differential privacy analytics API",
    'long_description': '# Tumult Analytics\n\nTumult Analytics is a library that allows users to execute differentially private operations on\ndata without having to worry about the privacy implementation, which is handled\nautomatically by the API. It is built atop the [Tumult Core library](https://gitlab.com/tumult-labs/core).\n\n## Installation\n\nSee the [installation instructions in the documentation](https://docs.tmlt.dev/analytics/latest/installation.html#prerequisites)\nfor information about setting up prerequisites such as Spark.\n\nOnce the prerequisites are installed, you can install Tumult Analytics using [pip](https://pypi.org/project.pip).\n\n```bash\npip install tmlt.analytics\n```\n\n## Documentation\n\nThe full documentation is located at https://docs.tmlt.dev/analytics/latest/.\n\n## Support\n\nIf you have any questions/concerns, please [create an issue](https://gitlab.com/tumult-labs/analytics/-/issues) or reach out to [support@tmlt.io](mailto:support@tmlt.io)\n\n## Contributing\n\nWe are not yet accepting external contributions, but please let us know at [support@tmlt.io](mailto:support@tmlt.io) if you are interested in contributing.\n\nSee [CONTRIBUTING.md](CONTRIBUTING.md) for information about installing our development dependencies and running tests.\n\n## License\n\nCopyright Tumult Labs 2022\n\nThe Tumult Platform source code is licensed under the Apache License, version 2.0 (Apache-2.0).\nThe Tumult Platform documentation is licensed under\nCreative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).\n',
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://www.tmlt.dev/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<3.10.0',
}


setup(**setup_kwargs)
