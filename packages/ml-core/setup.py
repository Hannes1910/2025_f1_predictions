from setuptools import setup, find_packages

setup(
    name="f1-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastf1>=3.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.8",
)