"""pytest configuration — ensures sibling modules (features.py, filter.py,
data.py, etc.) are importable without forcing tests to be run from inside
this directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
