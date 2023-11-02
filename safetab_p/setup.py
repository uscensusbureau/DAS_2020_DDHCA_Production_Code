# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.safetab_p']

package_data = \
{'': ['*'],
 'tmlt.safetab_p': ['resources/config/input/*',
                    'resources/config/output/*']}

install_requires = \
['tmlt.common>=0.8.0,<0.9.0',
 'tmlt.safetab_utils>=0.5.0,<1.0.0',
 'tmlt.analytics>=0.5.0,<1.0.0',
  'pandas>=1.2.0,<2.0.0',
  'pyspark[sql]==3.0.3',
  'numpy>=1.21.0,<1.21.6',
  'smart_open>=5.2.0,<6.0.0']

setup_kwargs = {
    'name': 'tmlt-safetabp',
    'version': '5.0.0',
    'description': 'SafeTab-P',
    'long_description': "# SafeTab-P Package",
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
