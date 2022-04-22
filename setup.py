from setuptools import setup

setup(
    name='data_manager',
    version='0.0.0.1',
    packages=['manager'],
    url='',
    license='',
    author='bjahnke',
    author_email='bjahnke71@gmail.com',
    description='manages saving/loading data to specified location and data formatting',
    install_requires=[
        'numpy',
        'pandas',
        'xlsxwriter',
        'matplotlib',
        'yfinance',
        'pandas_accessors @ git+https://github.com/bjahnke/pandas_accessors.git#egg=pandas_accessors'
    ]
)
