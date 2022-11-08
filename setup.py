from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

required_packages = [
    "torch", "flwr", "pytorch-lightning", "numpy", "scikit-learn"
]

setup(
    name="lightningflower",
    version="0.1.8",
    description="Pre-packaged federated learning framework using Flower and PyTorch-Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelRoeder/lightningflower",
    author="Manuel Roeder",
    author_email="manuel.roeder@web.de",
    license="MIT",
    packages=["lightningflower"],
    install_requires=required_packages,
    python_requires='>=3.8.12',
    package_data={"": ["README.md", "LICENSE"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
