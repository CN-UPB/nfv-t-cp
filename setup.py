"""
Copyright (c) 2017 Manuel Peuster
ALL RIGHTS RESERVED.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Manuel Peuster, Paderborn University, manuel@peuster.de
"""
from setuptools import setup, find_packages

setup(name='nfvppsim',
      version='0.0.1',
      license='Apache 2.0',
      description='NFV performance profiling simulation framework.',
      url='https://github.com/mpeuster/nfv-pp-sim',
      author_email='manuel@peuster.de',
      package_dir={'': '.'},
      packages=find_packages('.'),
      # include_package_data=True,
      # package_data={},
      install_requires=[
          "pyyaml",
          "numpy",
          "scipy",
          "simpy",
          "sklearn",
          "coloredlogs"
      ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'nfvppsim=nfvppsim:main',
          ],
      },
      setup_requires=[],
      tests_require=["pytest"],
      )
