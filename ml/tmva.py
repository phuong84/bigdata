#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     tmva.py

##
# This module contains functions to work with TMVA package

import logging
import yaml
from stringparser import *
from rootutils import *
from ROOT import *

##
# @class MyTMVA
# @brief This class
class MyTMVA:

    ## Dictionary for TMVA types
    _method = {'BayesClassifier': TMVA.Types.kBayesClassifier, 'Boost': TMVA.Types.kBoost, 'BDT': TMVA.Types.kBDT, 'DT': TMVA.Types.kDT, 'FDA': TMVA.Types.kFDA, 'Fisher': TMVA.Types.kFisher, 'LD': TMVA.Types.kLD, 'Likelihood': TMVA.Types.kLikelihood, ' KNN': TMVA.Types.kKNN, 'MLP': TMVA.Types.kMLP, 'SVM': TMVA.Types.kSVM}

    ## @brief The constructor
    # @param configfile name of configuration file to assign tasks to MyTMVA object
    def __init__(self, configfile=None):
        ## Logger object
        self._logger = logging.getLogger(self.__class__.__name__)
        ## List of tasks
        self._tasks = None
        if configfile:
            self.read_config(configfile)
    
    ## @brief The destructor
    def __del__(self):
        pass

    ## @brief Method to read configuration file
    # @param configfile name of configuration file to assign tasks to MyTMVA object
    def read_config(self, configfile=None):
        try:
            with open(configfile) as f:
                try:
                    config_dict = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        self._logger.error("Error in configuration file at position (line=%s,col=%s)" % (mark.line+1, mark.column+1))
                    return
        except IOError:
            self._logger.error('Could not open file '+configfile)
            return
        self._tasks = config_dict['TMVA']
        self._logger.debug('List of TMVA tasks %s'%(self._tasks))

    ## @brief Method to add task
    # @param task list of task parameters
    def add_task(self, task=None):
        if not task:
            self._logger.error('No task to be added')
            return
        self._tasks.append(task)

    ## @brief Method to process task 
    # @param taskid task ID (starting from 0)
    def process(self, taskid=0):
        if taskid < 0 or taskid > len(self._tasks)-1:
            self._logger.error('Invalid taskid = %d'%(taskid))
            return
        self._process(self._tasks[taskid])

    ## @brief Method to process all tasks
    def process_all(self):
        for task in self._tasks:
            self._process(task)
    
    ## @brief Method to categorize tasks and process them
    # @param task list of task parameters
    def _process(self, task=None):
        if not task:
            self._logger.error('No task to be processed')
            return
        taskname = task.get('Task',None)
        if taskname == 'TRAINING':
            self._training(task)
        elif taskname == 'VALIDATING':
            self._validating(task)
        elif taskname == 'PREDICTING':
            self._predicting(task)
        else:
            self._logger.error('Unknown task name '+taskname)

    ## @brief Method to process training task
    # @param task list of task parameters, the options for factory (FactOpt),
    # method (MethodOpt) and data preparation (PrepDataOpt) can be found at
    # http://tmva.sourceforge.net/old_site/optionRef.html
    def _training(self, task):
        filepath = task.get('Input','input.root')
        treename = task.get('Tree','tree')
        # read input tree
        tree = Tree(filepath,treename,True)
        # create output file
        outfile = TFile(get_path(filepath)+'/'+get_filename(filepath)+'_tmva.root','recreate')
        # create TMVA factory
        method = task.get('Method','BDT')
        self._logger.info('Starting TMVA - '+method)
        TMVA.Tools.Instance()
        factory = TMVA.Factory('TMVA',outfile,task.get('FactOpt',''))
        # add variables
        for x in task.get('XVars',[]):
            factory.AddVariable(x,'F')
        # specify signal and background
        factory.AddSignalTree(tree.get_tree())
        factory.AddBackgroundTree(tree.get_tree())
        sig_cut = TCut(task.get('SigCut',''))
        bkg_cut = TCut(task.get('BkgCut',''))
        # prepare the training/testing
        factory.PrepareTrainingAndTestTree(sig_cut,bkg_cut,task.get('DataPrepOpt',''))
        # book the method and train/test
        method = factory.BookMethod(self._method[method],method,task.get('MethodOpt',''))
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        outfile.Close()

    ## @brief Method to validate training task
    # @param task list of task parameters
    def _validating(self, task):
        files = []
        file_ = task.get('Input',None)
        if isinstance(file_,str):
            files.append(file_)
        elif isinstance(file_,list):
            files = file_
        method = task.get('Method','BDT')
        plots = task.get('PlotType',None)
        varlist = task.get('XVars',[])
        ext = task.get('FileExt','png')
        outdir = task.get('OutputDir','.')
        for f in files:
            self._logger.debug('Creating plots from TMVA output file '+f)
            for p in plots:
                if p == 'CorrMatrix':
                    self._make_corrmatrix_plot(f,outdir,ext)
                elif p == 'Variables':
                    self._make_vars_plot(f,method,outpath=outdir,extension=ext)
                elif p == 'OvertrainCheck':
                    self._make_overtrain_plot(f,method,outpath=outdir,extension=ext)
        if 'ROC' in plots:
            self._compare_ROC(files,method,outpath=outdir,extension=ext)

    ## @brief Method to process prediction task
    # @param task list of task parameters
    def _predicting(self, task):
        pass

    ## @brief Method to make correlation matrix plot
    # @param filepath path of input file
    # @param outpath directory where to put the output plots
    # @param extension output file extension, default is pdf 
    def _make_corrmatrix_plot(self, filepath, outpath='.', extension='pdf'):
        corrmatrix = ['CorrelationMatrixS','CorrelationMatrixB']
        f = TFile(filepath)
        for matrix in corrmatrix:
            histo = f.Get(matrix)
            plotter = Plotter(matrix)
            plotter.add_histo(histo,4,'matrix')
            plotter.plot_histos()
            self._logger.debug('Save file as '+outpath+'/'+get_filename(filepath)+'_'+matrix+'.'+extension)
            plotter.save_plot(outpath+'/'+get_filename(filepath)+'_'+matrix+'.'+extension)

    ## @brief Method to make signal/background variable comparison plots
    # @param filepath path of input file
    # @param method TMVA method for training
    # @param sig_title signal title
    # @param bkg_title background title
    # @param outpath directory where to put the output plots
    # @param extension output file extension, default is pdf
    def _make_vars_plot(self, filepath, method, sig_title='Signal', bkg_title='Background', outpath='.', extension='pdf'):
        name = get_filename(filepath)
        f = TFile(filepath)
        tree = f.Get('TrainTree')
        branches = [b.GetName() for b in tree.GetListOfBranches()]
        varlist = [branches[i] for i in range(2,len(branches)-2)]
        for var in varlist:
            signal = f.Get('Method_'+method+'/'+method+'/'+var+'__Signal')
            background = f.Get('Method_'+method+'/'+method+'/'+var+'__Background')
            if not signal or not background:
                self._logger.error('Cannot get signal or background histogram for variable '+var)
                continue
            signal.GetXaxis().SetTitle(var)
            signal.GetYaxis().SetTitle('Normalized')
            signal.SetTitle(sig_title)
            background.SetTitle(bkg_title)
            plotter = Plotter(var)
            plotter.add_histo(signal,2,'tmva_train')
            plotter.add_histo(background,4,'tmva_train')
            plotter.plot_histos(norm=True)
            plotter.build_legend(0.7,0.72,0.87,0.88)
            self._logger.debug('Save file as '+outpath+'/'+name+'_'+var+'.'+extension)
            plotter.save_plot(outpath+'/'+name+'_'+var+'.'+extension)

    ## @brief Method to make overtrain check plot
    # @param filepath path of input file
    # @param method TMVA method for training
    # @param sig_title signal title
    # @param bkg_title background title
    # @param outpath directory where to put the output plots
    # @param extension output file extension, default is pdf
    def _make_overtrain_plot(self, filepath, method, sig_title='Signal', bkg_title='Background', outpath='.', extension='pdf'):
        name = get_filename(filepath)
        f = TFile(filepath)
        # create train and test histograms
        train_signal     = TH1D('train_signal', sig_title+' (Train)', 40, -1.0, 1.0)
        train_background = TH1D('train_background', bkg_title+' (Train)', 40, -1.0, 1.0)
        test_signal      = TH1D('test_signal', sig_title+' (Test)', 40, -1.0, 1.0)
        test_background  = TH1D('test_background', bkg_title+' (Test)', 40, -1.0, 1.0)
        train_signal.Sumw2()
        train_background.Sumw2()
        test_signal.Sumw2()
        test_background.Sumw2()
        # project the histograms
        train_tree = f.Get('TrainTree')
        test_tree = f.Get('TestTree')
        signalCut = 'classID==0'
        backgroundCut = 'classID>0'
        train_tree.Project('train_signal',method,signalCut)
        train_tree.Project('train_background',method,backgroundCut)
        test_tree.Project('test_signal',method,signalCut)
        test_tree.Project('test_background',method,backgroundCut)
        # make overtrain plot
        train_signal.GetXaxis().SetTitle(method+' response')
        train_signal.GetYaxis().SetTitle('Normalized')
        plotter = Plotter('overtrain')
        plotter.add_histo(train_signal,2,'tmva_train')
        plotter.add_histo(train_background,4,'tmva_train')
        plotter.add_histo(test_signal,2,'tmva_test')
        plotter.add_histo(test_background,4,'tmva_test')
        plotter.plot_histos(norm=True)
        plotter.build_legend(0.72, 0.72, 0.87, 0.88)
        #text=TLatex()
        #text.SetNDC()
        kS = train_signal.KolmogorovTest(test_signal)
        kB = train_background.KolmogorovTest(test_background)
        result_string = 'KS-test for %s(%s): %.2f (%.2f)' %(sig_title,bkg_title,kS,kB)
        #text.DrawLatex(0.15,0.93,result_string)
        self._logger.debug('Save file as '+outpath+'/'+name+'_overtraining.'+extension)
        plotter.save_plot(outpath+'/'+name+'_overtraining.'+extension)
        

    ## @brief Method to make ROC plot
    # @param files list of input files
    # @param method TMVA method for training
    # @param outpath directory where to put the output plots
    # @param extension output file extension, default is pdf
    def _compare_ROC(self, files, method, outpath='.', extension='pdf'):
        plotter = Plotter()
        for i,filepath in enumerate(files):
            f = TFile(filepath)
            dirname = get_path(filepath)
            name = get_filename(filepath)
            histo = f.Get('Method_'+method+'/'+method+'/MVA_'+method+'_trainingRejBvsS')
            histo.SetDirectory(0)
            histo.SetTitle(method)
            plotter.add_histo(histo,i+4,'line')
            f.Close()
        plotter.plot_histos()
        self._logger.debug('Save file as '+outpath+'/ROC.'+extension)
        plotter.save_plot(outpath+'/ROC.'+extension)





