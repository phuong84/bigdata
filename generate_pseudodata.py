#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     generate_pseudodata.py

import os
import argparse
import logging
import logging.config
from stringparser import *
from pseudodata import *

## Main function
# @brief Read configuration file and parse the information in it
# @param args list of arguments provided
def main(args):
    # read data from input file
    data = PseudoData(args.inputfile, args.headerline, args.datatype, args.method)
    # benign data
    benign = data.filter_data(colid=11, value=2)
    # malignant data
    malignant = data.filter_data(colid=11, value=4)
    # calculate benign/data ratio
    ratio = float(len(benign))/len(data.get_data())
    # generate pseudo-data for benign and malignant
    if args.method != 'bootstrap':
        logger.error('Invalid method '+args.method)
        return
    data.book_method(args.method)
    nbenign = int(args.ndata*ratio)
    pseudo_benign = data.generate(benign, nbenign, args.size)
    pseudo_malignant = data.generate(malignant, args.ndata-nbenign, args.size)
    # write out pseudo-data
    pseudodata = pseudo_benign + pseudo_malignant
    data.save_as(pseudodata, args.outdir+'/breast-cancer-wisconsin_pseudodata.data', True)


if __name__ == "__main__":
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Generating pseudo-data from real data')
    parser.add_argument('inputfile', type=str, default='input.data', help='input file of real data')
    parser.add_argument('--datatype', type=str, default='int', help='type of input data')
    parser.add_argument('--headerline', type=bool, default=True, help='headerline included in data file')
    parser.add_argument('--outdir', type=str, default='.', help='directory which contains output files')
    parser.add_argument('--method', type=str, default='bootstrap', help='method for generating pseudo-data')
    parser.add_argument('--ndata', type=int, default=1000, help='number of pseudo-data')
    parser.add_argument('--size', type=int, default=2, help='size of sampled data')
    args = parser.parse_args()
    main(args)

