#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SELFPLAYTV SDK: make a team, train it on your own hardware, watch it play.

Deliberately empty of imports. sdk.train has to set NUMBA_NUM_THREADS before
anything from engine/ or trainer/ is imported, because numba sizes its thread
pool once, when it is first imported. A package that imported the engine here
would have fixed the thread count by the time `python -m sdk.train` got to read
--threads.
"""
