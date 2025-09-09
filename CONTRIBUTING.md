# Introduction

Welcome, and thank you for considering contributing to PyART! ✨

This document is meant to help make the development of PyART straightforward.
Here, you will find a set of best practices that we strive to follow (at times,
admittedly, failing). As such, they are meant to be guidelines, more than
strict rules.

## How to contribute

We welcome feedback, documentation, tutorials, bug reports, feature requests
as well as pull requests. Given the (lack of) personpower, we may not be able to get to your request
or provide a bugfix in short times. Please be patient!

### The issue tracker
Bug reports and feature requests should use the [issue tracker]().
Please do not file issues to ask questions regarding code usage.

### Code changes
If you wish to submit changes to the main code, you will have to do so via a
dedicated pull request. The easiest way to proceed is for you to create your own **fork** of
PyART and work on a dedicated branch. Once you are done:
- rebase your branch on the latest main (if needed)
- open a new pull request to PyART/main
- assign one of the main developers of the code to review your changes.

### Important notes
- We will only merge well-documented, commented code
- We will only merge code for which all automatic [tests](https://github.com/RoxGamba/PyART/tree/main/tests) pass
- If significant feaures are added or the logic of the code is modified, we will only merge code that is covered by tests.
- We try to format our code according to `black`. In order for this to be automatically done, we have a pre-commit hook that can be added via
  ```bash
  git config --local core.hooksPath hooks
  ```
