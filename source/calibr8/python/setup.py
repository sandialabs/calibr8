import numpy as np
from setuptools import setup, find_packages

setup(
    name="calibr8",
    version="1.0.0",
    description="Python tools for Calibr8",
    python_requires=">=3.8",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
    include_dirs=[np.get_include()],
    setup_requires=["numpy", "scipy"],
    install_requires=[
        "pycompadre"
    ],
    entry_points={
        "console_scripts": [
            "python_inverse=calibr8.bin.inverse:main"
        ]
    }
)
