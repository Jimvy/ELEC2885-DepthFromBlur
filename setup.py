from setuptools import setup

setup(
    name='Lazybot',
    packages=['Lazybot'],
    include_package_data=True,
    install_requires=[
        'numpy', 'astropy', 'matplotlib', 'scipy',
        'future', 'imageio', 'cython',  'pyfftw',
        'scikit-image', 'pillow', 'sporco'
    ],
)