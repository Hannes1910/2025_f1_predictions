from setuptools import setup, find_packages

setup(
    name="f1-ml-core",
    version="1.0.0",
    description="F1 Predictions ML Core Package",
    packages=find_packages(),
    install_requires=[
        "fastf1",
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "matplotlib",
        "seaborn",
        "requests",
    ],
    python_requires=">=3.8",
)