# Contributing

## Introduction

First off, thank you for considering contributions. We value community contributions!

## Contributions We Need

You may already know what you want to contribute \-- a fix for a bug you
encountered, or a new feature your team wants to use.

If you don't know what to contribute, keep an open mind! Improving
documentation, bug triaging, and writing tutorials are all examples of
helpful contributions that mean less work for you.

## Your First Contribution

Unsure where to begin contributing? You can start by looking through some of our issues [listed here](https://github.com/RedisVentures/redisvl/issues).

## Getting Started

Here's how to get started with your code contribution:

1.  Create your own fork of this repo
2.  Set up your developer environment
2.  Apply the changes in your forked codebase / environment
4.  If you like the change and think the project could use it, send us a
    pull request.

### Dev Environment
There is a provided `requirements.txt` and `requirements-dev.txt` file you can use to install required libraries with `pip` into your virtual environment.

Or use the local package editable install method:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all,dev]'
```

Then to deactivate the env:
```
source deactivate
```

### Linting and Tests

Check formatting, linting, and typing:
```bash
make check
```

Tests (with vectorizers):
```bash
make test-cov
```

Tests w/out vectorizers:
```bash
SKIP_VECTORIZERS=true make test-cov
```

> Dev requirements are needed here to be able to run tests and linting.
> See other commands in the [Makefile](Makefile)

### Docker Tips

Make sure to have [Redis](https://redis.io) accessible with Search & Query features enabled on [Redis Cloud](https://redis.com/try-free) or locally in docker with [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/):

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This will also spin up the [FREE RedisInsight GUI](https://redis.com/redis-enterprise/redis-insight/) at `http://localhost:8001`.

## How to Report a Bug

### Security Vulnerabilities

**NOTE**: If you find a security vulnerability, do NOT open an issue.
Email [Redis OSS (<oss@redis.com>)](mailto:oss@redis.com) instead.

In order to determine whether you are dealing with a security issue, ask
yourself these two questions:

-   Can I access something that's not mine, or something I shouldn't
    have access to?
-   Can I disable something for other people?

If the answer to either of those two questions are *yes*, then you're
probably dealing with a security issue. Note that even if you answer
*no*  to both questions, you may still be dealing with a security
issue, so if you're unsure, just email us.

### Everything Else

When filing an issue, make sure to answer these five questions:

1.  What version of python are you using?
2.  What version of `redis` and `redisvl` are you using?
3.  What did you do?
4.  What did you expect to see?
5.  What did you see instead?

## How to Suggest a Feature or Enhancement

If you'd like to contribute a new feature, make sure you check our
issue list to see if someone has already proposed it. Work may already
be under way on the feature you want -- or we may have rejected a
feature like it already.

If you don't see anything, open a new issue that describes the feature
you would like and how it should work.

## Code Review Process

The core team looks at Pull Requests on a regular basis. We will give
feedback as as soon as possible. After feedback, we expect a response
within two weeks. After that time, we may close your PR if it isn't
showing any activity.
