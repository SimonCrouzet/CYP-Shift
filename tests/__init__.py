"""Test package.

This file exists so ``tests.reference.official_metric`` resolves regardless of how
pytest is invoked. Without it, ``python -m pytest`` works (the current directory lands
on ``sys.path``) while a bare ``pytest`` does not, which is exactly the difference
between a local run and continuous integration.
"""
