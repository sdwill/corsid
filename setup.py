from setuptools import setup, find_packages

setup(
    name='corosid',
    version='0.1',
    description='System identification algorithms for space-based exoplanet coronagraphy',
    author='Scott Will',
    author_email='sdwill94@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'astropy',
        'jax',
        'svgutils',
        'svglib',
        'asdf'
 ]
)
