from setuptools import setup, find_packages

setup(
    name="enhanced-qml-pv-forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pennylane>=0.32.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    author="Mayan",
    author_email="mayan25sharma@gmail.com",
    description="Enhanced Quantum Machine Learning for PV Power Forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mayan10/Enhanced-HQLSTM-Model",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 