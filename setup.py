from setuptools import setup, find_packages

setup(name='turbulence-stats-ptv-tools',
      version='0.1',
      packages=find_packages(),
      author='Yaxing Li',
      url=r'https://github.com/YaxingLiZJU/turbulence-stats-ptv/',
      description='Python tools for process PTV trajectory data and obtain basic and advanced turbulence statistics',
      long_description=open('README.md').read(),
      install_requires=[
          'numpy',
          'scipy',
          ],
      )
