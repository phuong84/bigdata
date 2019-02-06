#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     tree.py

##
# This module contains functions for making, converting and comparing
# between different ROOT Tree.

import csv
import logging
from array import array
from stringparser import *
from plot import *
from ROOT import *

##
# @class Tree
# @brief This class
class Tree:

    ## @brief The constructor
    # @param tree input ROOT tree, data file of data list which the constructor will use to create internal ROOT tree
    # @param treename name of the internal ROOT tree, default is 'tree'
    # @param headerline can be boolean (if the input is ROOT tree) or list (if the input is data file or list)
    def __init__(self, tree=None, treename='tree', headerline=None):
        ## Logger object
        self._logger = logging.getLogger(self.__class__.__name__)
        if not tree:
            self._logger.warning('No tree specified')
            return
        if isinstance(tree,str):
            if get_extension(tree) == '.root':
                self.read_tree(tree, treename)
            elif tree.find('.') > -1:
                self.read_csv(tree, treename, headerline)
        elif isinstance(tree,list):
            self.read_data(tree, treename, headerline)
        else:
            ## TTree object
            self._tree = tree
        ## tree name
        self._treename = treename
        ## number of entries
        self._nentries = self.get_nentries()
        
    ## @brief The destructor
    def __del__(self):
        self._tree = None
    
    ## @brief Method to get number of entries
    # @return number of entries
    def get_nentries(self):
        if not self._tree:
            self._logger.error('No tree exists')
            return 0
        return self._tree.GetEntriesFast()

    ## @brief Method to get ROOT tree
    # @return ROOT tree
    def get_tree(self):
        if not self._tree:
            self._logger.error('No tree exists')
            return
        return self._tree
    
    ## @brief Method to get tree name
    # @return tree name
    def get_treename(self):
        return self._treename

    ## @brief Method to get list of branches
    # @return list of branches
    def get_branchlist(self):
        if not self._tree:
            self._logger.error('No tree exists')
            return
        branches = [b.GetName() for b in self._tree.GetListOfBranches()]
        return branches

    ## @brief Method to get histogram
    # @param branchname name of the branch which gives histogram
    # @param nbins number of bins in the histogram
    # @param min_ minimum value of the histogram
    # @param max_ maximum value of the histogram
    # @return histogram
    def get_histo(self, branchname='', nbins=None, min_=None, max_=None):
        tree = self._tree.Clone()
        if not min_:
            min_ = tree.GetMinimum(branchname)
        if not max_:
            max_ = tree.GetMaximum(branchname)
        if max_ <= min_:
            self._logger.error('Histogram has invalid (max <= min) values: max = %f, min=%f'%(max_,min_))
            return
        if not nbins:
            nbins = int(max_ - min_)
        self._logger.debug('%s: nbins = %d, min = %f, max=%f'%(branchname,nbins,min_,max_))
        hist = TH1F(self._treename+'_'+branchname, '', nbins, min_, max_)
        tree.Project(self._treename+'_'+branchname,branchname)
        if not hist:
            self._logger.debug('Could not retrieve histogram from branch '+branchname)
            return None
        return hist

    ## @brief Method to get ROOT tree from an ROOT file
    # @param filepath path of the input ROOT file
    # @param treename name of the created tree
    def read_tree(self, filepath='tree.root', treename='tree'):
        if not isinstance(filepath,str) or not isinstance(treename,str):
            self._logger.error('Invalid filepath or treename')
            return
        file_ = TFile(filepath)
        self._tree = file_.Get(treename)
        self._treename = treename
        self._nentries = self.get_nentries()

    ## @brief Method to read and convert CSV file to ROOT tree
    # @param filepath path of the input file
    # @param treename name of the created tree
    # @param headerline can be boolean (if the header is in input file) or list (if no header is in input file)
    def read_csv(self, filepath='tree.csv', treename='tree', headerline=None):
        if not isinstance(filepath,str) or not isinstance(treename,str):
            self._logger.error('Invalid filepath or treename')
            return
        filename = get_filename(filepath)
        self._tree = TTree(treename, filename)
        # read data from input file and fill to tree
        with open(filepath) as f:
            self._logger.debug('Opening ' + filepath)
            reader = csv.reader(f)
            header = None
            if isinstance(headerline,list):
                header = headerline
            elif isinstance(headerline,bool) and headerline == True:
                header = reader.next()
            else:
                self._logger.error('No header specified')
                return
            self._logger.debug('List of headers ['+' '.join(header)+']')
            var = []
            nvars = len(header)
            for i in range(nvars):
                a = array( 'f', [0.] )
                var.append(a)
                self._tree.Branch(header[i], var[i], header[i]+'/F')
            for row in reader:
                for i in range(nvars):
                    var[i][0] = str2float(row[i])
                self._tree.Fill()
            self._treename = treename
            self._nentries = self.get_nentries()

    ## @brief Method to read and convert data list to ROOT tree
    # @param data input data list
    # @param treename name of the created tree
    # @param header header list of the data
    def read_data(self, data=None, treename='tree', header=None):
        if not isinstance(data,list) or not isinstance(header,list) or len(data[0]) != len(header):
            self._logger.error('Invalid data or header')
            return
        if not isinstance(treename,str):
            self._logger.error('Invalid treename')
            return
        self._logger.debug('List of headers ['+' '.join(header)+']')
        self._tree = TTree(treename)
        var = []
        nvars = len(header)
        for i in range(nvars):
            a = array( 'f', [0.] )
            var.append(a)
            self._tree.Branch(header[i], var[i], header[i]+'/F')
        for row in data:
            for i in range(nvars):
                var[i][0] = str2float(row[i])
            self._tree.Fill()
        self._treename = treename
        self._nentries = self.get_nentries()

    ## @brief Method to write ROOT tree to file
    # @param filepath name of the output file
    def write_tree(self, filepath='tree.root'):
        rootfile = TFile(filename+'.root', 'recreate')
        rootfile.Write()
        rootfile.Close()
        logger.info(filename + '.root was created')
    
    ## @brief Method to compare between variables of internal ROOT tree and another ROOT tree
    # @param tree Tree object
    # @param varname name of a single variable (string) or list of variables
    def compare_tree(self, tree=None, varname=None):
        if not tree or not self._tree:
            self._logger.error('No tree for comparison')
            return
        if not varname:
            self._logger.error('No variable for comparison')
            return
        # create a list of variables
        varlist = []
        if isinstance(varname,str):
            varlist.append(varname)
        elif isinstance(varname,list):
            varlist = varname
        else:
            self._logger.debug('Invalid variable name')
            return
        # make comparison plot
        plotter = Plotter()
        for var in varlist:
            hist1 = self.get_histo(var)
            hist2 = tree.get_histo(var)
            self._logger.debug('%s: %s, mean = %f, rms=%f'%(self._treename,var,hist1.GetMean(),hist1.GetRMS()))
            self._logger.debug('%s: %s, mean = %f, rms=%f'%(tree.get_treename(),var,hist2.GetMean(),hist2.GetRMS()))
            plotter.plot_histos([hist1,hist2])
            plotter.save_plot('compare_'+var+'.pdf')
