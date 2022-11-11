from setuptools import setup

# function to read in requirements.txt to into a python list
def read_requirements():
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    return requirements

def read_dev_requirements():
    with open("requirements-dev.txt") as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name="redisvl",
    description="Vector loading utility for Redis vector search",
    license="BSD-3-Clause",
    version="0.1.0",
    python_requires=">=3.6",
    install_requires=read_requirements(),
    extras_require={"dev": read_dev_requirements()},
    packages=["redisvl"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "redisvl = redisvl.cli.__init__:main"
        ]
    }
)