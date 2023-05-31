# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given (apropos credit, this file is largely inspired from https://github.com/audreyfeldroy/cookiecutter-pypackage/blob/master/CONTRIBUTING.rst, thanks for the amazing work!).

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Fraunhofer-IESE/badgers/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Badgers could always use more documentation, whether as part of the
official Badgers docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at URL.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `badgers` for local development.

1. Fork the `badgers` repo on GitHub.
2. Clone your fork locally

```
    $ git clone https://github.com/Fraunhofer-IESE/badgers
```

3. Install dependencies and start your virtualenv:

```
    $ pip install 
```

5. Create a branch for local development:

```
    $ git checkout -b name-of-your-bugfix-or-feature
```

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

```
    $ tox
```

7. Commit your changes and push your branch to GitHub:

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python PYTHON-VERSION and for PyPy. Check
   https://github.com/Fraunhofer-IESE/badgers and make sure that the tests pass for all supported Python versions.


