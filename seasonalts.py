'''
Created on 14 feb 2012

@author: thomasg

'''
#
#imports
#import os
#from os import listdir
#from os.path import isfile, join
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.stats import norm, mstats, stats
from scipy.signal import correlate,convolve2d, convolve
#from scipy.signal import convolve
#from scipy.stats.stats.pearsonr import pearsonr
import pandas as pd
#from seasonal import fit_seasons, adjust_seasons
import statsmodels.api as sm
from pandas import Series
import matplotlib.pyplot as pyplot


#import mj_datetime_v64 as mj_dt

import math

from seasonal import fit_seasons, adjust_seasons
#import seasonal

#import pylab as pl


#...your code...

#pyplot.ioff()
#pyplot.show()
           
def MKtest(x):  
    """   
    Input:
        x:   a vector of data
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        z: normalized test statistics 
    """
    n = len(x)
    # calculate S 
    s = 0
    for k in range(n-1):
        #for j in range(k+1,n):           
        #    s += np.sign(x[j] - x[k])
        t = x[k+1:]
        u = t - x[k]
        sx = np.sign(u)
        s += sx.sum()
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    return z

def TheilSenXY(x,y):  
    res = mstats.theilslopes(x,y)
    return res[0][0],res[1][0]

def Autocorrelate(sig,nlags,partial):
    if partial:
        return sm.tsa.stattools.pacf(sig,nlags=nlags,alpha=.05)
    return sm.tsa.stattools.acf(sig,nlags=nlags,alpha=.05)

def CrossCorrelate(i,j,method = 'fft', maxlag = 0): 
    if maxlag == 0:
        SNULLE
        return correlate(i, j)
    else:
        BULLE
        return correlate(i, j)[len(j)-maxlag-1:len(j)+maxlag]
    
    
def AutoCorrelation(x, maxlag):
    """
    Autocorrelation with a maximum number of lags.

    `x` must be a one-dimensional numpy array.

    This computes the same result as
        numpy.correlate(x, x, mode='full')[len(x)-1:len(x)+maxlag]

    The return value has length maxlag + 1.
    """
    #x = _check_arg(x, 'x')
    p = np.pad(x.conj(), maxlag, mode='constant')
    T = as_strided(p[maxlag:], shape=(maxlag+1, len(x) + maxlag),
                   strides=(-p.strides[0], p.strides[0]))
    return T.dot(p[maxlag:].conj())


      
def CrossCorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return value has length 2*maxlag + 1.
    """
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')

    return T.dot(px)  

def PearsonNrScipy(x,y):
    return stats.pearsonr(x,y)[0]

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def PearsonNrOld(x, y):
    n = len(x)
    assert n > 0
    avg_x = float(sum(x)) / n
    avg_y = float(sum(y)) / n
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    if diffprod <= 0:
        return 0 
    return diffprod / math.sqrt(xdiff2 * ydiff2)

def PearsonNr(x, y):
    from geoimagine.ktnumba import PearsonNrNumba
    return PearsonNrNumba(x,y)
    '''
    n = len(x)
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    xydiff2 = xdiff2 * ydiff2
    if xydiff2 <= 0:
        return 0
    return diffprod / math.sqrt(xydiff2)
    '''

def PearsonNrAlt(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(map(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

class GraphPlot():
    def __init__(self):
        pass
    
    def GraphDecomposeMKTSIni(self,TS,mk,slope,intercept,figtitle):
        self.ts = TS.ts
        tsr = []
        for i in range(len(TS.ts)):
            tsr.append(intercept+slope*i)
        regrA = np.array(tsr)
        title = 'Timeseries decomposition (type= %(d)s; period = %(p)d; smoothing = %(t)s)' %{'d':TS.decompose, 'p':TS.period, 't':TS.trend}
        if TS.decompose == 'trend': seastitle = 'Seasons (trend: %s)' %(TS.trend)
        elif TS.decompose == 'cyclic': seastitle = 'Seasons (trend: spline)'
        elif TS.decompose == 'naive': seastitle = 'Seasons (trend: naive)'
        elif TS.decompose == 'detrend': 
            if TS.prefilterseason:
                seastitle = 'Seasons (trend: spline)'
            else:
                seastitle = 'Seasons (trend: %s)' %(TS.trend)
        trendtitle = 'Trend (MKtest Z = %(mk)1.2f; regression = Theil Sen)' %{'i':TS.ts, 'mk':mk}

        self.GraphDecomposeMKTSR(TS,title,trendtitle,seastitle,regrA,TS.decompose,figtitle)
          
    def GraphSeasonal(self,TS, title, normalize=False, npA=False, labelA=False,npB=False, labelB=False, ts=False, tendency=False, seasons=False, adjusted=False, residual=False, longtendency=False , detrended=False): 
        pyplot.figure()
        if ts:
            if normalize:
                normA  = (TS.ts - mean(TS.ts)) /  std(TS.ts)
                pyplot.plot(normA, label='data')
            else:
                #plt.plot(data.index, data.amount)
                #pyplot.plot(TS.ts.index, TS.ts.data.amount)
                pyplot.plot(TS.ts, label='data')
        if tendency:
            if normalize:
                normA  = (self.tendency - np.average(self.tendency)) /  np.std(self.tendency)
                pyplot.plot(normA, label='trend')
            else:
                
                pyplot.plot(self.tendency, label='trend')
        if detrended:
            if normalize:
                normA  = (self.detrended - np.average(self.detrended)) /  np.std(self.tdetrended)
                pyplot.plot(normA, label='detrended')
            else:          
                pyplot.plot(self.detrended, label='detrended')
        if seasons:
            if normalize:
                normA  = (self.seasons - np.average(self.seasons)) /  np.std(self.seasons)
                pyplot.plot(normA, label='data')
            else:
                pyplot.plot(self.seasons, label='seasons')   
        if adjusted:
            pyplot.plot(self.adjusted, label='adjusted')
        if residual:
            pyplot.plot(self.residual, label='residual')
        if longtendency:
            if normalize:
                normA  = (TS.longtendency - np.average(TS.longtendency)) /  np.std(TS.longtendency)
                pyplot.plot(normA, label='data')
            else:
                pyplot.plot(TS.longtendency, label='long term tendency')
        if labelA:
            pyplot.plot(npA, label=labelA)
        if labelB:
            pyplot.plot(npB, label=labelB)
        if normalize:
            title = '%(t)s (normalized)' %{'t':title}
        pyplot.suptitle(title)
        pyplot.legend(loc='upper left')
        pyplot.show()
        
    def GraphPandas(self,TS, title, normalize=False, npA=False, labelA=False,npB=False, labelB=False, ts=False, tendency=False, seasons=False, adjusted=False, residual=False, longtendency=False , detrended=False): 
        pyplot.figure()
        if ts:
            if normalize:
                normA  = (TS.df - mean(TS.df)) /  std(TS.df)
                pyplot.plot(normA, label='data')
            else:
                pyplot.plot(TS.df.index, TS.df.values)
        if tendency:
            if normalize:
                normA  = (self.tendency - np.average(self.tendency)) /  np.std(self.tendency)
                pyplot.plot(normA, label='trend')
            else:
                
                pyplot.plot(TS.tendency, label='trend')
        if detrended:
            if normalize:
                normA  = (self.detrended - np.average(self.detrended)) /  np.std(self.tdetrended)
                pyplot.plot(normA, label='detrended')
            else:          
                pyplot.plot(self.detrended, label='detrended')
        if seasons:
            if normalize:
                normA  = (self.seasons - np.average(self.seasons)) /  np.std(self.seasons)
                pyplot.plot(normA, label='data')
            else:
                pyplot.plot(self.seasons, label='seasons')   
        if adjusted:
            pyplot.plot(self.adjusted, label='adjusted')
        if residual:
            pyplot.plot(self.residual, label='residual')
        if longtendency:
            if normalize:
                normA  = (TS.longtendency - np.average(TS.longtendency)) /  np.std(TS.longtendency)
                pyplot.plot(normA, label='data')
            else:
                pyplot.plot(TS.longtendency, label='long term tendency')
        if labelA:
            pyplot.plot(npA.df.index, npA.df.values)
            #pyplot.plot(npA, label=labelA)
        if labelB:
            pyplot.plot(npB.df.index, npB.df.values)
            #pyplot.plot(npB, label=labelB)
        if normalize:
            title = '%(t)s (normalized)' %{'t':title}
        pyplot.suptitle(title)
        pyplot.legend(loc='upper left')
        pyplot.show()
             
    def GraphDecomposeMKTSR(self,TS,title,trendtitle,seastitle,tsr,decomp,figtitle):
        f, (ax1, ax2, ax3, ax4) = pyplot.subplots(4, sharex=True)
        if decomp == 'detrend':
            ax1.plot(TS.ts, label='detrended data')
        else:         
            ax1.plot(TS.ts, label='data')
        ax1.set_title(title)
        ax1.legend(loc='best')
        ax2.plot(TS.fullseasons, label='seasons')
        ax2.set_title(seastitle)
        ax2.legend(loc='best')
        ax3.plot(TS.residual, label='residual')
        ax3.set_title('residual')
        ax3.legend(loc='best')
        if decomp == 'detrend':
            ax4.plot(TS.longtendency, label='multi cycles trend')
        else:
            ax4.plot(TS.tendency, label='trend')
        ax4.plot(tsr)
        ax4.set_title(trendtitle)
        ax4.legend(loc='best')
        f.subplots_adjust(hspace=0.25)
        pyplot.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        f.canvas.set_window_title(figtitle)
        pyplot.show()
        
    def GraphDualDecomposeTrends(self,yearfac,title): 
        pyplot.figure() 
        pyplot.plot(self.trend, label='original trend')
        legstr = 'multiperiod trend (%s cycles)' %(yearfac)
        pyplot.plot(self.longtrend, label=legstr)
        pyplot.plot(self.stationary, label='stationary data')
        pyplot.plot(self.stationarytrend, label='stationary trend')
        pyplot.legend(loc='best')
        pyplot.suptitle(title)
        pyplot.show()
        
    def GraphCrossCorrTrends(self,y,x,title):
        pyplot.figure() 
        pyplot.plot(y, label='forcing')
        pyplot.plot(x, label='data')
        pyplot.legend(loc='best')
        pyplot.suptitle(title)
        pyplot.show()
        
    def GraphCrossCorrMultiTrends(self,dataD,figtitle):
        fig, ax = pyplot.subplots(len(dataD), 1, figsize=(8,6))
        for a,key in zip(ax,dataD.keys() ):
            y = dataD[key]['index']
            z = dataD[key]['item']
            n = len(y)
            x = np.linspace(1,n,n)
            a.plot(x,y)
            a.plot(x,z)
            title = 'crosscorr %(b)s vs %(i)s (lag = %(l)d; corr = %(c)1.2f; pearson = %(p)1.2f)' %{'b':dataD[key]['band'],'i':dataD[key]['i'],'l':dataD[key]['lag'],'c':dataD[key]['corr'], 'p':dataD[key]['pearson']}
            a.set_title(title)
            # add labels/titles and such here 
        fig.subplots_adjust(hspace=0.25)
        pyplot.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.canvas.set_window_title(figtitle)
        pyplot.show()
        
    def GraphMultiDecompIni(self,dataD,figtitle,tendency=False):
        if tendency:
            for D in dataD:
                tsr = []
                for i in range(len(dataD[D]['ts'])):   
                    tsr.append(dataD[D]['b']+dataD[D]['a']*i)
                    regrA = np.array(tsr)
                dataD[D]['tsr'] = regrA
            self.GraphMultiTrends(dataD,figtitle)   
        else:
            self.GraphMultiDecomp(dataD,figtitle)
            #self.GraphMultiDecompAllInOne(dataD,figtitle)
        
    def GraphMultiDecomp(self,dataD,figtitle):
        fig, ax = pyplot.subplots(len(dataD), 1, figsize=(8,8), sharex=True)

        for a,key in zip(ax,dataD.keys() ):
            x = dataD[key]['index'].index
            y = dataD[key]['index'].values
            #z = dataD[key]['item']
            #n = len(y)

            #x = np.linspace(1,n,n)
            a.plot(x,y)
            #a.plot(x,z)
            title = 'Decomposition method %(b)s' %{'b':dataD[key]['band']}
            a.set_title(title)
            # add labels/titles and such here 
        
        pyplot.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        #pyplot.setp(a.index, a.values, visible=False)
        #pyplot.plot(npB.df.index, npB.df.values)
        fig.canvas.set_window_title(figtitle)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25)
        pyplot.show()
        
    def GraphMultiDecompAllInOne(self,dataD,figtitle):
        fig, ax = pyplot.subplots(len(dataD), 1, figsize=(8,8), sharex=True)

        for a,key in zip(ax,dataD.keys() ):
            x = dataD[key]['index'].index
            y = dataD[key]['index'].values
            #z = dataD[key]['item']
            #n = len(y)

            #x = np.linspace(1,n,n)
            #a.plot(x,y)
            #a.plot(x,z)
            #title = 'Decomposition method %(b)s' %{'b':dataD[key]['band']}
            #a.set_title(title)
            # add labels/titles and such here 
            pyplot.plot(x, y)
        #pyplot.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        #pyplot.setp(a.index, a.values, visible=False)
        #pyplot.plot(npB.df.index, npB.df.values)
        fig.canvas.set_window_title(figtitle)
        fig.tight_layout()
        #fig.subplots_adjust(hspace=0.25)
        pyplot.show()
        
    def GraphMultiTrends(self,dataD,figtitle):
        fig, ax = pyplot.subplots(len(dataD), 1, figsize=(8,8), sharex=True)
        for a,key in zip(ax,dataD.keys() ):
            #y = dataD[key]['index']
            x = dataD[key]['index'].index
            y = dataD[key]['index'].values
            z = dataD[key]['tsr']
            #n = len(y)

            #x = np.linspace(1,n,n)
            a.plot(x,y)
            a.plot(x,z)
            title = 'Decomposition method %(b)s (mk = %(mk)1.2f; ts = %(ts)1.2f)' %{'b':dataD[key]['band'],'mk':dataD[key]['mk'],'ts':dataD[key]['a']}
            a.set_title(title)
            # add labels/titles and such here 
        #fig.subplots_adjust(hspace=0.1)
        pyplot.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.canvas.set_window_title(figtitle)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25)
        pyplot.show()
        
    def GraphCellSeasonIni(self, origdataD, seasonD, residualD, tendencyD, figtit):

        dataD = {'origdata':origdataD,'season':seasonD, 'residual':residualD, 'trend':tendencyD}
        self.GraphCellSeason(dataD,figtit)
        
    def GraphCellSeason(self,dataD,figtitle):
        fig, ax = pyplot.subplots(4, 1, figsize=(8,8), sharex=True)

        for a,key in zip(ax,dataD.keys() ):
            x = dataD[key]['index'].index
            y = dataD[key]['index'].values
            a.plot(x,y)
            #a.plot(x,z)
            if dataD[key]['comp'] == 'trend':
                title = 'Component: %(b)s (mk = %(mk)1.2f; ts = %(ts)1.2f)' %{'b':dataD[key]['band'],'mk':dataD[key]['mk'],'ts':dataD[key]['a']}

                tsr = []
                for i in range(len(dataD[key]['ts'])):   
                    tsr.append(dataD[key]['b']+dataD[key]['a']*i)
                    regrA = np.array(tsr)
                #dataD[D]['tsr'] = regrA
                a.plot(x,regrA)
            else:
                title = 'Component: %(b)s' %{'b':dataD[key]['comp']}
            a.set_title(title)
            # add labels/titles and such here 
        
        pyplot.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        #pyplot.setp(a.index, a.values, visible=False)
        #pyplot.plot(npB.df.index, npB.df.values)
        fig.canvas.set_window_title(figtitle)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25)
        pyplot.show()
        
class TScommon:
    def __init__(self, period, params): 
        
        if params.additive:
            model = 'additive'
        else:
            model = 'multiplicative'
        self.model = model
        self.forceseason =  params.forceseason
        self.prefilterseason = params.prefilterseason
        self.trend = params.trend
        self.yearfac = params.yearfac

        if params.trend == 'kernel': 
            if len(params.kernel) > 5 and ',' in params.kernel:
                self.kernel = np.array([float(x) for x in params.kernel.split(',')])
                self.kernel /= self.kernel.sum()
                self.halfperiod = int(math.ceil(self.kernel.shape[0]/2))    
            else:
                exitstr = 'Something wrong with the kernel' 
                exit(exitstr)
            self.period = period.pdTS.annualperiods
        else:
            if params.kernel != '0':
                print ('Ignoring kernel, you have to set trend to kernel to make the kernel take effect')
            self.kernel = False
            
            #set halfperiod to half the annual observations
            self.period = period.pdTS.annualperiods
            self.halfperiod = int(math.ceil(period.pdTS.annualperiods/2)) 
        self.kernelcutstart = self.halfperiod
        self.kernelcutend = len(period.datumL)-self.halfperiod

        if self.yearfac > 1 and self.kernel:
            exit('When using a kernel, the yearfac should be unity (1)')
        if params.maxlag < 0:
            self.maxlag = self.halfperiod
        else:
            self.maxlag = params.maxlag

        self.lagadjust = np.arange(-self.maxlag, self.maxlag+1)

            
    def _PearsonRoll(self,lag,npA,npB):
        if lag >= 0: 
            #Roll to fit the seasons
            c_sig = np.roll(npA, shift=int(np.ceil(lag)))
            #pearsonnr = PearsonNrScipy(c_sig[lag:],npB[lag:])
            pearsonnr = PearsonNr(c_sig[lag:],npB[lag:])
            return (lag, pearsonnr, c_sig[lag:], npB[lag:])
        else:
            lag *= -1
            #Roll to fit the seasons
            c_sig = np.roll(npB, shift=int(np.ceil(lag)))
            #pearsonnr = PearsonNrScipy(npA[lag:],c_sig[lag:])
            pearsonnr = PearsonNr(npA[lag:],c_sig[lag:])
            return -lag, pearsonnr, npA[lag:], c_sig[lag:]
                                            
    def CrossCorr(self, npA, npB, mirrorlag, maxlag): 
        '''Cross correlation with maximum lag set 
        '''
        if mirrorlag:
            crossCorr = CrossCorrelation(npB, npA, self.maxlag)
            lag0 = np.argmax(crossCorr)
            lag = self.lagadjust[lag0] 

        else:            
            crossCorr = CrossCorrelation(npB, npA, self.maxlag)[self.maxlag:]
            lag = np.argmax(crossCorr) 

        return self._PearsonRoll(lag,npA,npB) 
      
    def CrossCorrAbs(self, npA, npB, mirrorlag, maxlag): 
        '''Cross correlation identifying the absolute largest correlation  
        '''
        if mirrorlag:
            crossCorr = np.absolute(CrossCorrelation(npB, npA, self.maxlag))
            lag0 = np.argmax(crossCorr)
            lag = self.lagadjust[lag0]    
        else:      
            crossCorr = np.absolute(CrossCorrelation(npB, npA, self.maxlag)[self.maxlag:])
            lag = np.argmax(crossCorr)
            
        return self._PearsonRoll(lag,npA,npB)
   
    def _CrossCorrFixedLag(self, npA, npB, lag): 
        '''
        '''   
        return self._PearsonRoll(lag,npA,npB)[1]
        
    def MannKenndalTheilSen(self,ts): 
        mkz = MKtest(ts)
        slope,intercept = TheilSenXY(ts,self.trendy)
        return mkz,slope,intercept
    
    def ConvolutionFilter(self,ts):
        y = convolve(ts, self.kernel, mode='valid')
        y = np.concatenate([ts[0:self.ke0],y,  ts[self.kb1:self.ke1]  ])
        return y
             
class NaiveTS(TScommon):
    def __init__(self, period, params):
        TScommon.__init__(self, period, params)
        #Preset the kernel, to save time and to handle multuyear trends
        if not self.kernel:
            freq = self.period*self.yearfac
            if freq % 2 == 0:  # split weights at ends
                filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
            else:
                filt = np.repeat(1./freq, freq)
            self.kernel = filt
                
    def SetTS(self,ts):
        self.ts = ts
        
    def SetDF(self,df):
        self.df = df

    def SeasonalDecompose(self):
        #The kernel is always prefixed
        self.res = sm.tsa.seasonal_decompose(self.ts, model=self.model, 
                    filt=self.kernel, freq=self.period, extrapolate_trend=1)
        self.window = self.kernel.shape[0]

        self.tendency = self.res.trend
        self.fullseasons = self.res.seasonal  
        self.residual = self.res.resid
        self.seasons = self.fullseasons 
        
class SeasonalTS(TScommon):
    
    def __init__(self, period, params):
        TScommon.__init__(self, period, params)
               
    def SetTS(self,ts):
        self.ts = ts       
    
    def SeasonalDecompose(self):
        self.seasons, self.tendency, self.window = fit_seasons(self.ts, trend=self.trend, period=self.period, ptimes=self.yearfac, 
                                                               splineseason=self.prefilterseason, forceseason=self.forceseason, kernel=self.kernel)
        self.detrended = self.ts-self.tendency
        if self.seasons is None:
            SNULLEBULLE
            self.adjusted = self.residual = self.detrendednoise = self.trendseasonal = None
        else:
            self.adjusted = adjust_seasons(self.ts, seasons=self.seasons, period=self.period)
            self.residual = self.adjusted - self.tendency
            self.fullseasons = self.ts-self.tendency-self.residual 
                        
class PandasTimeSeries(TScommon):
    def __init__(self, period, params):
        TScommon.__init__(self, period, params)
            
    def SetTS(self,ts):
        self.ts = ts
        self.CreatePandasDF()
            
    def CreatePandasDF(self):
        self.df = Series(self.ts, index=self.origdatearr)
        
    def CreatePandasTrendDF(self,ts):
        self.df = Series(ts, index=self.trenddatearr)
        
    def CreatePandasLongTrendDF(self,ts):
        self.df = Series(ts, index=self.longtrenddatearr)
        
    def CreatePandasSeasonalDF(self,TS):
        self.ts = Series(TS.ts, index=self.origdatearr)
        self.fullseasons = Series(TS.fullseasons, index=self.origdatearr)
        self.tendency = Series(TS.tendency, index=self.origdatearr)
        self.residual = Series(TS.residual, index=self.origdatearr)
        if hasattr(TS,'longtendency'):
            self.longtendency = Series(TS.longtendency, index=self.origdatearr)
            
    def CreatePandasTSDF(self,TS):
        if len(TS.ts) == len(self.origdatearr):
            self.ts = Series(TS.ts, index=self.origdatearr)
        elif len(TS.ts) == len(self.trenddatearr):
            self.ts = Series(TS.ts, index=self.trenddatearr)
        elif len(TS.ts) == len(self.longtrenddatearr):
            self.ts = Series(TS.ts, index=self.longtrenddatearr)
        else:
            BALLE

            #cut the ends:
            toomany = len(self.origdatearr)-len(TS.ts)
            left = int(toomany/2)
            rigth = toomany-left
            right = len(self.origdatearr)-rigth          
            newDate = self.origdatearr[left:right]
            self.ts = Series(TS.ts, index=newDate)

        if len(TS.fullseasons) == len(self.origdatearr):
            self.fullseasons = Series(TS.fullseasons, index=self.origdatearr)
        elif len(TS.fullseasons) == len(self.trenddatearr):
            self.fullseasons = Series(TS.fullseasons, index=self.trenddatearr)
        else:
            toomany = len(self.origdatearr)-len(TS.fullseasons)
            left = int(toomany/2)
            rigth = toomany-left
            right = len(self.origdatearr)-rigth
            newDate = self.origdatearr[left:right]
            self.fullseasons = Series(TS.fullseasons, index=newDate)
            
        toomany = len(self.origdatearr)-len(TS.tendency)
        left = int(toomany/2)
        rigth = toomany-left
        right = len(self.origdatearr)-rigth
        newDate = self.origdatearr[left:right]
        self.tendency = Series(TS.tendency, index=newDate)
        #self.tendency = Series(TS.tendency, index=self.trenddatearr)
        self.residual = Series(TS.residual, index=newDate)
        if hasattr(TS,'longtendency'):
            toomany = len(self.origdatearr)-len(TS.longtendency)
            left = int(toomany/2)
            rigth = toomany-left
            right = len(self.origdatearr)-rigth
            newDate = self.origdatearr[left:right]
            self.longtendency = Series(TS.longtendency, index=newDate)
            #self.longtrendts = Series(TS.longtrendts, index=newDate)
        
            
            
    def CreatePandasNoneTSDF(self,TS,period):
        self.origdatearr = pd.date_range(period.startdate, period.enddate, freq='AS')
        self.ts = Series(TS.ts, index=self.origdatearr)
        self.tendency = Series(TS.ts, index=self.origdatearr)
            
    def ResampleToAnnualSum(self):
        return self.df.resample('AS', how=sum)
    
    def ResampleToAnnualAvg(self):
        return self.df.resample('AS')
     
    def ResampleToMonthAvg(self):
        return self.df.groupby(self.df.index.month).mean()
                  
class PandasSimple:
    def __init__(self, period):
        self.SetDates(period)
        
    def SetDates(self,period):
        self.origdatearr = pd.date_range(period.startdate, period.enddate, freq='M')
        
    def CreatePandasDF(self, ts):
        self.df = Series(ts, index=self.origdatearr)
            
def GraphAcfBoxWhisker(data,title,xList):
    halflife = 1.33 #the influence is reduced by half for each month
    add = 1-halflife
    ewmlL = []
    zL = []

    for i in xList:
        #z = add+halflife*math.pow(halflife,-i/halflife)
        #y = -0.25+1.25*math.exp(-i/halflife)
        y = -0.25+1.25*math.exp(-i/halflife)
        #z = add+halflife*math.pow(halflife,-i/halflife)
        #z = halflife*math.exp(-i/halflife)
        #z = -0.25+1.25*math.pow(2,-i/halflife)
        z = -0.25+1.25*math.pow(2,-i/halflife)
        ewmlL.append(y)
        zL.append(z)   
    #trend = np.array([1,0.8,0.6,0.4,0.3,0.2,0.1])
    fig = pyplot.figure()
    #pyplot.boxplot(data, 0, '')
    pyplot.boxplot(data)
    pyplot.plot(ewmlL, label='EWML')
    pyplot.plot(zL, label='zL')
    #pyplot.plot(medianL, label='median')
    #pyplot.plot(meanL, label='mean')

    pyplot.ylabel('correlation')
    pyplot.xlabel('lag (months)')  
    #props = dict(color='black', linewidth=2, markeredgewidth=2)
    #make_xaxis(fig, 0, offset=-1, **props)
    #pyplot.xlim(0, 12)
    pyplot.xlim(xmin=0)
    fig.suptitle(title) 
    pyplot.show()
    BALLE
