import subprocess


def format():
    subprocess.run(["isort", "./redisvl", "./tests/", "--profile", "black"], check=True)
    subprocess.run(["black", "./redisvl", "./tests/"], check=True)


def check_format():
    subprocess.run(["black", "--check", "./redisvl"], check=True)


def sort_imports():
    subprocess.run(["isort", "./redisvl", "./tests/", "--profile", "black"], check=True)


def check_sort_imports():
    subprocess.run(
        ["isort", "./redisvl", "--check-only", "--profile", "black"], check=True
    )


def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "./redisvl"], check=True)


def check_mypy():
    subprocess.run(["python", "-m", "mypy", "./redisvl"], check=True)


def test():
    subprocess.run(["python", "-m", "pytest", "-n", "6", "--log-level=CRITICAL"], check=True)


def test_verbose():
    subprocess.run(
        ["python", "-m", "pytest", "-n", "auto", "-vv", "-s", "--log-level=CRITICAL"], check=True
    )


def test_notebooks():
    subprocess.run(["cd", "docs/", "&&", "poetry run treon", "-v"], check=True)


def build_docs():
    subprocess.run("cd docs/ && make html", shell=True)


def serve_docs():
    subprocess.run("cd docs/_build/html && python -m http.server", shell=True)
