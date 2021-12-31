from setuptools import setup

required_packages = [
    "torch", "flwr", "pytorch-lightning", "numpy"
]

setup(
    name="lightningflower",
    version="0.1.0",
    description="Flower based Pytorch lightning framework",
    url="https://github.com/ManuelRoeder/lightningflower",
    author="Manuel Roeder",
    author_email="manuel.roeder@web.de",
    license="MIT",
    packages=["lightningflower"],
    install_requires=required_packages,
    python_requires='>=3.8.12',
    package_data={"": ["README.md", "LICENSE"]},
)
