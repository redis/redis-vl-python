import subprocess
import sys


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
    test_cmd = ["python", "-m", "pytest", "-n", "auto", "--log-level=CRITICAL"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def test_verbose():
    test_cmd = [
        "python",
        "-m",
        "pytest",
        "-n",
        "auto",
        "-vv",
        "-s",
        "--log-level=CRITICAL",
    ]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def test_notebooks():
    test_cmd = [
        "python",
        "-m",
        "pytest",
        "--nbval-lax",
        "./docs/user_guide",
        "-vvv",
    ]
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)

    subprocess.run(
        test_cmd,
        check=True,
    )


def build_docs():
    subprocess.run("cd docs/ && make html", shell=True)


def serve_docs():
    subprocess.run("cd docs/_build/html && python -m http.server", shell=True)
