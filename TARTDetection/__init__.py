#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : __init__.py.py

__version__ = '0.0.2'


def version():
    return __version__


import importlib

module_spec = importlib.util.find_spec('TART')
assert module_spec is not None
