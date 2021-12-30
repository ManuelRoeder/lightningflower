from setuptools import setup

setup(
    name='lightningflower',
    version='0.1.0',
    description='Flower based Pytorch lightning framework',
    url='https://github.com/ManuelRoeder/lightningflower',
    author='Manuel Roeder',
    author_email='manuel.roeder@fhws.de',
    license='MIT',
    packages=['lightningflower'],
    install_requires=['flwr>=0.17.0',
                      'pytorch-lightning>=1.5.4',
                      'numpy>=1.21.4',
                      ],
)
