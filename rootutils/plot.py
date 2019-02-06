#!/usr/bin/env python

##
# @author   Dang Nguyen Phuong (dnphuong1984@gmail.com)
# @date     30-01-2019
# @file     plot.py

##
# This module contains functions for plotting features
# in ROOT file.

import logging
from stringparser import *
from ROOT import * 

##
# @class Plotter
# @brief This class
class Plotter:

    ## @brief The constructor
    # @param name name of the internal canvas
    # @param xsize width of the canvas
    # @param ysize height of the canvas
    def __init__(self, name='', xsize=700, ysize=500):
        ## Logger object
        self._logger = logging.getLogger(__name__)
        ## TCanvas object
        self._canvas = TCanvas(name, '', 0, 0, xsize, ysize)
        ## List of histograms
        self._histos = []
        self._legend = None

    ## @brief The destructor
    def __del__(self):
        pass

    ## @brief Method to set name and size to the canvas
    # @param name of the canvas
    # @param xsize width of the canvas
    # @param ysize height of the canvas
    def set_canvas(self, name='', xsize=700, ysize=500):
        self._canvas.SetName(name)
        self._canvas.SetCanvasSize(xsize,ysize)

    '''
    def set_label(self, n_channel, xtitle):
        histo = TH2F("","", n_channel, -1., 3.5, n_channel+1, -0.5, n_channel)
        histo.GetYaxis().SetLabelOffset(0.04)
        histo.GetYaxis().SetNdivisions(0)
        histo.SetXTitle(xtitle)
        text = TLatex()
        text.SetTextAlign(32)
        text.SetTextSize(0.03)
        return histo, text
    '''

    ## @brief Method to add histogram
    # @param histo histogram object
    # @param color color of histogram
    # @param style plot style of histogram
    # @param title title of histogram
    def add_histo(self, histo=None, color=0, style='line', title=None):
        opt = self._style[style]
        if histo:
            histo.SetLineColor(color)
            histo.SetLineWidth(opt[0])
            histo.SetLineStyle(opt[1])
            histo.SetMarkerColor(color)
            histo.SetMarkerSize(opt[2])
            histo.SetMarkerStyle(opt[3])
            histo.SetFillStyle(opt[4])
            if opt[4]:
                histo.SetFillColor(color)
            if title:
                histo.SetTitle(title)
            self._histos.append(histo)

    ## @brief Method to make plot from histograms
    # @param histos list of histograms (if not define, the internal histogram will be used)
    # @param xtitle title of x-axis
    # @param ytitle title of y-axis
    # @param plotstyle option for plotting histograms, default is 'hist'
    # @param norm option to normalize histograms, default is False
    # @param smooth option to smooth histograms, default is False
    def plot_histos(self, histos=None, xtitle='', ytitle='', plotstyle='hist', norm=False, smooth=False):
        self._set_plotstyle()
        self._canvas.cd()
        if not histos:
            histos = self._histos
        if not histos:
            return
        for hist in histos:
            if not hist or hist.Integral() == 0.:
                self._logger.error('Could not read histogram')
                return
            if norm:
                hist.Scale(1./hist.Integral())
            if smooth:
                hist.Smooth()
        isFirst = True
        for hist in histos:
            if isFirst:
                hist.SetStats(kFALSE)
                hist.GetXaxis().SetTitle(xtitle)
                hist.GetYaxis().SetTitle(ytitle)
                ymax = max([h.GetMaximum() for h in histos])
                ymax *= 1.2
                hist.SetMaximum(ymax)
                if norm: hist.SetMaximum(ymax/hist.Integral())
                hist.Draw(plotstyle)
                isFirst = False
            else:
                hist.Draw(plotstyle+' same')

    ## @brief Method to create a TGraph object
    # @param n number of points
    # @param x list of x values
    # @param y list of y values
    # @param e1,e2,e3,e4 list of x and y value errors
    # @param color color of the graph
    # @param style plotting style of the graph
    # @return TGraph object
    def create_graph(self, n, x, y, e1=None, e2=None, e3=None, e4=None, color=0, style='line'):
        zeros = array("d", [0]*n)
        if not e1:
            graph = TGraph(n, x, y)
        elif not e2:
            graph = TGraphErrors(n, x, y, e1, zeros)
        elif not e3:
            graph = TGraphErrors(n, x, y, e1, e2)
        elif not e4:
            graph = TGraphAsymmErrors(n, x, y, e1, e2, e3, zeros)
        else:
            graph = TGraphAsymmErrors(n, x, y, e1, e2, e3, e4)
        opt = self._style[style]
        graph.SetLineColor(color)
        graph.SetLineWidth(opt[0])
        graph.SetLineStyle(opt[1])
        graph.SetMarkerColor(color)
        graph.SetMarkerSize(opt[2])
        graph.SetMarkerStyle(opt[3])
        return graph

    ## @brief Method to draw a legend
    # @param showStat option to show mean and rms values of histograms
    # @param x1,y1,x2,y2 size of the legend
    # @param style plotting style of the legend
    # @param tsize size of legend text
    def add_legend(self, showStat=False, x1=0.75, y1=0.8, x2=0.9, y2=0.9, style='l', tsize=0.02):
        self._legend = TLegend(x1, y1, x2, y2)
        self._legend.SetBorderSize(0)
        self._legend.SetFillColor(0)
        self._legend.SetTextSize(tsize)
        for hist in self._histos:
            if hist:
                string = hist.GetTitle()
                if showStat:
                    string = '%s(%.2f,%.2f)' % (hist.GetTitle(),hist.GetMean(),hist.GetRMS())
                self._legend.AddEntry(hist, string, style)
        self._legend.Draw()

    ## @brief Method to quick draw a legend
    # @param x1,y1,x2,y2 size of the legend
    def build_legend(self, x1=0.75, y1=0.8, x2=0.9, y2=0.9):
        self._canvas.cd(1).BuildLegend(x1, y1, x2, y2).SetFillColor(0)
        
    ## @brief Method to save plot to a file
    # @param filename name of output file
    def save_plot(self, filename='plot.pdf'):
        self._canvas.SaveAs(filename)

    ## @brief Method to set plotting style
    # @param isLog option to make log scale, default is False
    # @param isGrid option to make grid, default is False
    def _set_plotstyle(self, isLog=False, isGrid=False):
        gROOT.SetBatch(True)
        gROOT.ProcessLine('gErrorIgnoreLevel = kWarning;')
        gStyle.SetOptTitle(0)
        if isLog:
            gPad.SetLogy()
            gPad.RedrawAxis()
        if isGrid:
            self._canvas.SetGrid()

    ## Dictionary of histogram plotting style,
    # the order of ist item is
    #[linewidth,linestyle,markersize,markerstyle,fillstyle,plotstyle]
    _style = {
        'line': [1,1,0,0,0,'l'],
        'boldline': [2,1,0,0,0,'l'],
        'dashline': [1,9,0,0,0,'l'],
        'square': [0,0,2,21,0,'p'],
        'dot': [0,0,2,20,0,'p'],
        'square_err': [1,1,2,21,0,'ep'],
        'dot_err': [1,1,2,20,0,'ep'],
        'hist': [1,1,0,0,0,'hist'],
        'solidhist': [1,1,0,0,1001,'hist'],
        'tmva_train': [1,1,0,0,3001,'hist'],
        'tmva_test': [1,1,1,20,0,'ep'],
        'matrix': [0,0,0,0,1001,'col text']
        }
