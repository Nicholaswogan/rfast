from skbuild import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rfast",
    packages=['rfast'],
    python_requires='>=3.7',
    version="2.7.6",
    license="MIT",
    install_requires=['numpy>=1.21','numba','scipy','ruamel.yaml','astropy>=5.0','emcee>=3.1','dynesty','multiprocess'],
    author='Tyler Robinson',
    author_email = 'robinson.tyler.d@gmail.com',
    description = "Generates synthetic spectra of planets.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "",
    include_package_data=True,
    cmake_args=['-DSKBUILD=ON']
)


