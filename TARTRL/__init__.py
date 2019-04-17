#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: __init__.py.py
"""
__version__ = '0.0.1'

def version():
    return __version__


import importlib

module_spec = importlib.util.find_spec('TART')
assert module_spec is not None
