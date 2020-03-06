import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vimpy",
    version="2.0.1",
    author="Brian Williamson",
    author_email="brianw26@uw.edu",
    description="vimpy: nonparametric variable importance assessment in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bdwilliamson/vimpy",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
