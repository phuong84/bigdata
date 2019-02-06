#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     stringparser.py

## 
# This module contains functions for extracting information
# from a string.  

import os

## String to boolean conversion
# @param string input string
# @return a boolean
def str2bool(string):
    return v.lower() in ("yes", "true", "t", "1")

## String to integer conversion
# @param string input string
# @return an integer number
def str2int(string):
    try:
        val = int(string)
        return val
    except ValueError:
        return 0

## String to float conversion
# @param string input string
# @return a float number
def str2float(string):
    try:
        val = float(string)
        return val
    except ValueError:
        return float('nan')

## Get path of directory which contain the input file 
# @param filepath input file path
# @return path of directory
def get_path(filepath):
    dirname, filename = os.path.split(filepath)
    if dirname =='':
        dirname = '.'
    return dirname

## Get full input file name from a given path
# @param filepath input file path
# @return full input file name
def get_fullfilename(filepath):
    dirname, filename = os.path.split(filepath)
    return filename

## Get input file name from a given path
# @param filepath input file path
# @return input file name
def get_filename(filepath):
    dirname, filename = os.path.split(filepath)
    return os.path.splitext(filename)[0]

## Get extension of input file name from a given path
# @param filepath input file path
# @return input file extension
def get_extension(filepath):
    dirname, filename = os.path.split(filepath)
    return os.path.splitext(filename)[1]
