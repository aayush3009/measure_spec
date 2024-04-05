from setuptools import setup, find_packages

setup(
    name='measure_spec',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'specutils',
        'mpdaf',
        'pyneb',
        'dust_extinction',
        'scipy',
    ],
    author='Aayush Saxena',
    author_email='aayush.saxena@physics.ox.ac.uk',
    description='A Python package for analyzing JWST spectra, but applies to all spectra in general.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aayush3009/measure_spec',
    license='MIT',
)