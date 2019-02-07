#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     31-01-2019
# @file     pseudodata.py

##
# This module contains functions for generating pseudo-data
# from real data or distribution.

import csv
import logging
import random
from stringparser import *

##
# @class PseudoData
# @brief This class 
class PseudoData:

    ## @brief The constructor
    # @param inputfile input file of real data
    # @param headerline True if there is header line in the input file, default is False
    # @param datatype type of input data (either 'int' or 'float'), default is 'int'
    # @param method method for pseudo-data generation, default is 'bootstrap'
    def __init__(self, inputfile=None, headerline=False, datatype='int', method='bootstrap'):
        ## Logger object
        self._logger = logging.getLogger(self.__class__.__name__)
        ## Real data
        self._data = None
        ## Pseudo-data
        self._pseudodata = None
        ## List of headers
        self._header = None
        ## Pseudo-data geeration method
        self._method = method if isinstance(method,str) else 'bootstrap'
        ## Type of input data
        self._type = datatype if isinstance(datatype,str) else 'int'
        if isinstance(inputfile,str):
            self.get(inputfile,headerline,datatype)
    
    ## @brief The destructor
    def __del__(self):
        self._data = None
        self._pseudodata = None

    ## @brief Method for getting input data
    # @param inputfile input file of real data
    # @param headerline True if there is header line in the input file, default is False
    # @param datatype type of input data (either 'int' or 'float'), default is 'int'
    def get(self, inputfile='', headerline=False, datatype='int'):
        self._logger.debug('Open file '+ inputfile)
        with open(inputfile) as f:
            reader = csv.reader(f)
            if headerline:
                self._header = reader.next()
            self._data = list(reader)
        self._type = datatype if isinstance(datatype,str) else 'int'

    ## @brief Method for filtering input data (by row)
    # @param data input data
    # @param colid column id (starting from 1) 
    # @param value data row will be selected if row[rowid] == value
    # @param outfile name of output file which the filtered data will be written to
    # @return filtered data (in list)
    def filter_data(self, data=None, colid=None, value=None, outfile=None):
        if not data:
            data = self._data
        if not data:
            self._logger.error('No data to be filtered')
            return
        if not isinstance(colid,int) or colid < 1 or colid > len(data) or not isinstance(value,(int,float)):
            self._logger.warning('Column-id or value is not correct. Cannot filter data')
            return data
        subdata = []
        for row in data:
            if float(row[colid-1]) == value:
                subdata.append(row)
        if not subdata:
            self._logger.warning('No data left after filtering, return full data instead')
            return data
        if len(self._header) == len(subdata[0]):
            self._logger.debug('Data was filtered with condition: '+self._header[colid-1]+' = %f'%value)
        if isinstance(outfile,str):
            with open(outfile,'w') as f:
                writer = csv.writer(f)
                writer.writerows(subdata)
        self._logger.debug('Filter ratio is %d'%(len(subdata))+'/%d'%(len(data)))
        return subdata

    ## @brief Method for getting real data
    # @return real data
    def get_data(self):
        return self._data

    ## @brief Method for getting pseudo-data
    # @return real data
    def get_pseudodata(self):
        return self._pseudodata

    ## @brief Method for getting header list
    # @return header list
    def get_header(self):
        return self._header
    
    ## @brief Method for setting header list
    # @param header list
    def set_header(self, header=[]):
        self._header = header if isinstance(header,list) else None

    ## @brief Method for setting data type
    # @param datatype type of input data (either 'int' or 'float'), default is 'int' 
    def set_datatype(self, datatype='int'):
        self._type = datatype if isinstance(datatype,str) else 'int'

    ## @brief Method for setting pseudo-data generation method
    # @param method method for pseudo-data generation, default is 'bootstrap'
    def book_method(self, method):
        self._method = method if isinstance(method,str) else 'bootstrap'

    ## @brief Method for generating pseudo-data
    # @param *opt parameters for pseudo-data generation, these parameters will be passed
    # to other methods (@_bootstrap)
    # @return pseudo-data
    def generate(self, *opt):
        if self._method == 'bootstrap':
            if len(opt) < 1:
                self._logger.error('Not enough parameter for '+self._method+' method')
                return
            self._bootstrap(*opt)
        if not self._pseudodata:
            self._logger.warning('No pseudo-data was generated')
            return
        return self._pseudodata
            
    ## @brief Method for writing pseudo-data to file
    # @param outfile name of output file which the real/pseudo-data will be written to
    # @param headerline True if the header line will be writen to output file, default is False
    def save_as(self, data=None, outfile='pseudodata.txt', headerline=False):
        if isinstance(data,str):
            outfile = data
            data = None
        if not data:
            self._logger.warning('No input data, going to write pseudo-data ...')
            data = self._pseudodata
        if not data:
            self._logger.warning('No pseudo-data to be written, going to write real data ...')
            data = self._data
        if not data:
            self._logger.error('No real data to be written')
            return
        with open(outfile,'w') as f:
            writer = csv.writer(f)
            if headerline:
                writer.writerow(self._header)
            writer.writerows(data)
        self._logger.info('Data was written to file '+outfile)

    ## @brief Method for generating pseudo-data by using bootstrap
    # @param *opt parameters for pseudo-data generation (input data, number of bootstrap samples,
    # bootstrap sample size)
    def _bootstrap(self, *opt):
        if isinstance(opt,(list,tuple)):
            data = opt[0]
            nsamples = opt[1] if len(opt) > 1 else 1000
            sample_size = opt[2] if len(opt) > 2 else 2
        else:
            data = self._data
            nsamples = opt[0] if len(opt) > 0 else 1000
            sample_size = opt[1] if len(opt) > 1 else 2
        self._logger.debug('Generate pseudo-data with '+self._method+' method, nsamples = %d'%(nsamples)+', sample size = %d'%(sample_size))
        self._pseudodata = []
        nrows = len(data)
        for _ in range(nsamples):
            sample = []
            for _ in range(sample_size):
                rowid = random.randrange(nrows)
                sample.append(data[rowid])
            avg = []
            for i in range(len(sample[0])):
                avg.append(self._get_row_average(sample,i))
            self._pseudodata.append(avg)

    ## @brief Method for calculating average (mean) value of each column
    # @param rows list of rows
    # @param col column id (starting from 0)
    # @return average (mean) value
    def _get_row_average(self, rows=None, col=None):
        sum_ = 0
        for row in rows:
            if self._type == 'int':
                sum_ += str2int(row[col])
            else:
                sum_ += str2float(row[col])
        avg = sum_/len(rows)
        return avg

