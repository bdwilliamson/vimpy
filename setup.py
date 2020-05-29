import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vimpy",
    version="2.0.2",
    author="Brian Williamson",
    author_email="brianw26@uw.edu",
    description="vimpy: perform inference on algorithm-agnostic variable importance in python",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bdwilliamson/vimpy",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers=(
        "Programming Language :: Python :: 3.3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
