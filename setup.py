from setuptools import setup
import setuptools
setup(
    name='fksplines',
    version='0.6',
    packages=setuptools.find_packages(),
    url='https://github.com/ged1182/splines',
    license='MIT License',
    author='George Dekermenjian',
    author_email='ged1182@gmail.com',
    description='A package to do uni-variate and multi-variate free-knot spline regression.',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires='>=3.7',
    long_description='A package to do uni-variate and multi-variate free-knot spline regression.'
)
