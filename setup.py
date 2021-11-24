from setuptools import setup
from src.version import __version__

setup(
    name='inflation_rnd',
    url='https://github.com/Piers14/inflation_rnd',
    author='Piers Hinds',
    author_email='pmxph7@nottingham.ac.uk',
    packages=['inflation_rnd'],
    install_requires=['numpy', 'pandas', 'scipy', 'rpy2'],
    tests_require=['pytest'],
    version=__version__,
    license='MIT',
    description='Methods to estimate the RND of inflation rates'
)