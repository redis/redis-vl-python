import subprocess

def format():
    subprocess.run(["isort", "./redisvl", "./tests/", "--profile", "black"])
    subprocess.run(["black", "./redisvl", "./tests/"])

def check_format():
    subprocess.run(["black", "--check", "./redisvl"])

def sort_imports():
    subprocess.run(["isort", "./redisvl", "./tests/", "--profile", "black"])

def check_sort_imports():
    subprocess.run(["isort", "./redisvl", "--check-only", "--profile", "black"])

def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "./redisvl"])

def mypy():
    subprocess.run(["mypy", "./redisvl"])

def test():
    subprocess.run(["python", "-m", "pytest", "--log-level=CRITICAL"])

def test_verbose():
    subprocess.run(["python", "-m", "pytest", "-vv", "-s", "--log-level=CRITICAL"])

def test_cov():
    subprocess.run(["python", "-m", "pytest", "-vv", "--cov=./redisvl", "--log-level=CRITICAL"])

def cov():
    subprocess.run(["coverage", "html"])
    print("If data was present, coverage report is in ./htmlcov/index.html")

def test_notebooks():
    subprocess.run(["cd", "docs/", "&&", "treon", "-v"])