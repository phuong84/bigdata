#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     dataframe.py

##
# This module contains functions for working with data frame.

import logging
import pandas as pd
import numpy as np
#import mathplotlib as plt

class DataFrame:

    def __init__(self, data=None, sep=',', delimiter=None, header=None, names=None):
        ## Logger object
        self._logger = logging.getLogger(self.__class__.__name__)
        self._df = None
        if isinstance(data,str):
            self.read_csv(data,sep,delimiter,header,names)
        if isinstance(data,str):
            self._df = pd.DataFrame(data=data)

    def __del__(self):
        pass

    def read_csv(self, data=None, sep=',', delimiter=None, header=None, names=None):
        if isinstance(data,str):
            self._data = pd.read_csv(data,sep,delimiter,header,names)
        return self._data

    def get_data(self):
        data = []
        for index, row in df.iterrows():
            mylist = [row.cluster, row.load_date, row.budget, row.actual, row.fixed_price]
            data.append(mylist)
        return data
