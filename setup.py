from setuptools import setup, find_packages

# function to read in requirements.txt to into a python list
def read_requirements():
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    return requirements

def read_dev_requirements():
    with open("requirements-dev.txt") as f:
        requirements = f.read().splitlines()
    return requirements

def read_all_requirements():
    with open("requirements-all.txt") as f:
        requirements = f.read().splitlines()
    return requirements

extras_require = {
    "all": read_all_requirements(),
    "dev": read_dev_requirements()
}


setup(
    name="redisvl",
    version="0.1.1",
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require=extras_require,
    packages=find_packages(),
    package_data={
        "redisvl": [
            "requirements.txt", "requirements-dev.txt", "requirements-all.txt"
        ]
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "rvl = redisvl.cli.runner:main"
        ]
    }
)
