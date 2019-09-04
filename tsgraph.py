'''
Created on 14 Dec 2018

@author: thomasgumbricht
'''

#import geoimagine.gis.mj_gis_v80 as mj_gis
#from geoimagine.support import karttur_dt as mj_dt
import numpy as np
from geoimagine.timeseries.timeseries import TScommon
from geoimagine.kartturmain.timestep import TimeSteps
from geoimagine.timeseries import seasonalts as mj_ts
from geoimagine.extract import extract
from geoimagine.ktgraphics import GraphPlot
import pandas as pd

class ProcessTimeSeriesGraph(TScommon):
    'class for modis specific processing'   
    def __init__(self, process, session, verbose):
        TScommon.__init__(self, process, session, verbose)

        print ('    starting ProcessTimeSeries:', self.process.proc.processid)
        
        if self.process.proc.processid[0:8].lower() == 'plotdbts':   
            self._indexGraph()
            return
        if self.process.proc.processid[0:12].lower() == 'autocorrdbts':
            self._indexAutoCorr()
            return
        if self.process.proc.processid[0:13].lower() == 'componentdbts':
            self._indexComponentGraphIni()
            return
        
        self._SetExtractFeatures()
        
        #Link each extract to a locus
        self._LinkExtractLocus()
        
        #Organize the timeseries data
        srccompL = self._CheckSrcTS() 

        self.srccomp = srccompL[0]

        self.srcDateD = {}
        for srccomp in srccompL:
            self.srcDateD[srccomp] = []
    
        self.srcCellNullD = {}
        self.geoFormatD = {}
        
        if 'xcross' in self.process.proc.processid and 'index' in self.process.proc.processid:
            #Read the indexes to use in the cross correlation
            self._indexComponentXcross()
                
        for locus in self.process.srcLayerD:
            if len(self.locusFeature[locus]) == 0:
                continue
            
            #open the input data
            for comp in srccompL:
                self._OpenSrcLayers(locus,comp)
                 
            self.srcDateL = self.srcDateD[self.srccomp]
                            
            if self.process.proc.processid[0:13].lower() == 'extractseason':
                self.dstArr = np.ones( ( len(self.dstDateL) ), np.float32)
                self.seasons = len(self.dstDateL)
                self._TSExtractSeason()
                        
            elif self.process.proc.processid.lower() in ['resampletsancillary','resampletssmap']:    
                self.dstArr = np.ones( ( len(self.dstDateL) ), np.float32)                 
                self.years = len(self.dstDateL)
                self.resampleperiods = self.process.dstperiod.periodstep

                if self.process.dstperiod.timestep[len(self.process.dstperiod.timestep)-1] in ['D','M'] :
                    #Create an array that stats the start and end columns to resample for each output
                    #print ('self.dstDateL',self.dstDateL)
                    self._TSresampleDstep()
                else:
                    self._TSresample()
    
            elif self.process.proc.processid.lower() == 'trendtsancillary':
                self.srccomp = srccompL[0]
                self.tsy = np.arange(len(self.process.srcperiod.datumL))
                self._TStrendIni()
                 
            elif self.process.proc.processid.lower() in ['seasonfilltsmodissingletile']:
                for comp in srccompL:
                    if comp == 'seasons':    
                        self.seasonKey = 'seasons'
                    else:
                        self.mainKey = comp
                self._TSSeasonFill()
   
            elif self.process.proc.processid.lower()[0:17] == 'indexcompxcrossts':
                self.srccomp = srccompL[0]
                self.stats = False
                self._compComponentXcross(locus)
                self._IndexCompXCrossGraph(locus)
                
            elif self.process.proc.processid.lower()[0:17] == 'layerxcompts':
                self.srccomp = srccompL[0]
                self.stats = False
                #self._compComponentXcross(locus)
                self._LayerXCorrGraph(locus)
                    
            elif self.process.proc.processid.lower()[0:15] == 'componentcompts':
                self.srccomp = srccompL[0]
                self.stats = False  
                
                self._compComponentGraphIni(locus)  
            elif self.process.proc.processid.lower()[0:7] == 'tsgraph':
                self.srccomp = srccompL[0]            
                self._tsGraphIni(locus)
                
            else:
                exitstr = 'Exiting, processid %(p)s missing in ProcessTimeSeriesGraph' %{'p':self.process.proc.processid}
                exit(exitstr)
            self._CloseSrcRasterFiles(locus)
    
    def _SetExtractFeatures(self):
        '''
        '''
        self.extractL = []
  
        if self.process.proc.processid[0:7] == 'tsgraph':
            #just a single position to extract
            plotidtxt = 'point'
            feature = {'type':'point','plotidtxt':plotidtxt, 'x':self.process.params.x, 'y':self.process.params.y}
            self.extractL.append(feature)
        elif self.process.proc.processid[0:15] == 'componentcompts':
            for item in self.process.proc.xy.paramsD:
                itemParams =  self.process.proc.xy.paramsD[item]
                feature = {'type':'point','id':itemParams['id'], 'x':itemParams['x'], 'y':itemParams['y'],'itemparams':itemParams}
                self.extractL.append(feature)
        elif self.process.proc.processid[0:17] == 'indexcompxcrossts':
            #Replica of the above
            for item in self.process.proc.xy.paramsD:
                itemParams =  self.process.proc.xy.paramsD[item]
                feature = {'type':'point','id':itemParams['id'], 'x':itemParams['x'], 'y':itemParams['y'],'itemparams':itemParams}
                self.extractL.append(feature)
            '''   
            feature = {'type':'point','id':plotidtxt, 'x':self.process.params.x, 'y':self.process.params.y}
            self.extractL.append(feature)
            '''
        else:
            print (self.process.proc.processid)
            ADDENDUMDUMKULA
            
    def _LinkExtractLocus(self):
        '''Links the extactvectors to the correct locus
        '''
        self.locusFeature = {}
        for locus in self.process.srcLayerD:
            #Get the extent of the locus
            locusD = self.process.srclocations.locusD[locus]
            paramL =['epsg','ullat','ullon','urlat','ullon','lrlon','lrlat','lllat','lllon','minx','miny','maxx','maxy']
            self.locusFeature[locus] = []
            if locusD['division'] == 'region':
                queryD = {'regionid':locus}
                
                rec = self.session._SelectRegionExtent(queryD,paramL)
                #Create a region and list all points that are in
                #Now I just put all points in
                for item in self.extractL:
                    self.locusFeature[locus].append(item)  
                                
    def _tsGraphIni(self,locus):
        '''
        '''
        for item in self.locusFeature[locus]:
            x,y, = item['x'],item['y']
            self._ExtractTimeSeriesPlot(locus,x,y)
            self.datumA = [self.process.srcperiod.datumD[datum]['acqdate'] for datum in self.process.srcLayerD[locus]] 
            self._tsGraph()
            '''
            data = {'date': datumA,
                    'ts': self.npA}
            self.df = pd.DataFrame(data, columns = ['date', 'ts'])
            graph = GraphPlot()
            graph._GraphSingleTimeLine(self.df,self.process.params)
            '''
            
    def _indexGraph(self):
        '''
        '''
        #Set the TS and decomposition
        self._SetTSDecomposition()
        data = {}
        for item in self.process.proc.index.paramsD: 
            self.npA = np.array( [x[1] for x in self.session._SelectClimateIndex(self.process.srcperiod,item)] )
            data[item] = self.npA
        self.srcperiod = TimeSteps(self.process.proc.srcperiodD)
        self.datumA = [self.srcperiod.datumD[datum]['acqdate'] for datum in self.srcperiod.datumL]
        data['dates'] = self.datumA
        
        self.df = pd.DataFrame(data, columns = list(data.keys()))
        self.df = self.df.set_index(pd.DatetimeIndex(self.df['dates']))
        
        graph = GraphPlot()
        if self.process.params.decompose:
            self.data={}
            self.regressD={}
            for item in self.process.proc.index.paramsD:
                self.TS.SetTS(self.df[item].values)
                self.data[item], self.regressD[item],titleAddendum = self._tsDecompose()
            self.process.params.title = '%(t)s: %(ta)s' %{'t':self.process.params.title,'ta':titleAddendum}
            if self.process.params.separate:
                graph._GraphMultiTimeLineDeCompose(self.data,self.regressD, self.process.proc.index.paramsD,self.process.params)
            else:
                #all in one graph
                graph._GraphMultiTimeLineDeComposeSingle(self.data,self.regressD, self.process.proc.index.paramsD,self.process.params)
                
        elif self.process.params.separate:
            graph._GraphMultiTimeLineSubPlots(self.df,self.process.proc.index.paramsD,self.process.params)
        elif self.process.params.pdplot:
            graph._PdGraph(self.df)
        else:  
            graph._GraphMultiTimeLine(self.df,self.process.proc.index.paramsD,self.process.params)

    def _indexAutoCorr(self):
        '''
        '''
        data = {}
        for item in self.process.proc.index.paramsD: 
            self.npA = np.array( [x[1] for x in self.session._SelectClimateIndex(self.process.srcperiod,item)] )
            if self.process.params.resampleseasonal:
                lagA,confintA = self._SeasonalAutoCorrFIX(self.npA,self.process.params.partial)
            else:
                lagA,confintA = self._FullAutoCorrGraph(self.npA,self.process.params.partial)
            lowlim = confintA[:, 0]            
            #hilim = confintA[:, 1]
            err = lagA-lowlim
            data[item] = {'acf':lagA,'err':err}
        xpos = np.arange(len(lagA))
        graph = GraphPlot()
        graph._BarChartSingle(xpos, data, self.process.proc.index.paramsD, self.process.params)
    
    def _FullAutoCorrGraph(self,ts,partial):
        '''
        '''
        #from statsmodels.tsa.stattools import acf, pacf
        from statsmodels import api as sm
        if np.isnan(np.sum(ts)):
            return False
        if partial:
            #I had to convert the numpy array to list, otherwise it did not work
            lagA,confintA = sm.tsa.pacf(ts.tolist(),nlags=self.process.params.nlags,alpha=.05)
        else:
            #I had to convert the numpy array to list, otherwise it did not work
            lagA,confintA = sm.tsa.acf(ts.tolist(),nlags=self.process.params.nlags,alpha=.05)

        return lagA,confintA

    def _compComponentGraphIni(self,locus):
        #Set the TS and decomposition
        obs = self._compComponentGraph(locus)
        self.layoutD = self.process.proc.xy.paramsD
        self._ComponentGraph(obs)
        
    def _indexComponentGraphIni(self):
        '''
        '''
        index = self._indexComponentGraph()
        self.layoutD = self.process.proc.index.paramsD
        self._ComponentGraph(index)
        
    def _indexComponentXcross(self):
        '''
        '''
        self.index = self._indexComponentGraph()
        self.indexLayoutD = self.process.proc.index.paramsD
        
    def _compComponentXcross(self,locus):
        '''
        '''
        self.obs = self._compComponentGraph(locus)
        self.layoutD = self.process.proc.index.paramsD

    def _indexComponentGraph(self):
        index = {}
        for item in self.process.proc.index.paramsD: 
            index[item] = np.array( [x[1] for x in self.session._SelectClimateIndex(self.process.srcperiod,item)] )
        self.srcperiod = TimeSteps(self.process.proc.srcperiodD)
        self.datumA = [self.srcperiod.datumD[datum]['acqdate'] for datum in self.srcperiod.datumL]
        return index
    
    def _IndexCompXCrossGraph(self,locus):
        '''
        '''
        if len(self.obs) > 1 and self.process.params.lagplot:
            exit('Plotting crosscorrelation with multiple samples requires that "lagplot" is set to False (default)')
        if len(self.obs) == 0 or len(self.index) == 0:
            exit('Plotting crosscorrelation requires at least one observation point and one index')
        xcrosscompsL = []
        if self.process.params.xcrossobserved:
            xcrosscompsL.append('observed')
        if self.process.params.xcrosstendency:
            xcrosscompsL.append('tendency')
        if self.process.params.xcrosseason:
            xcrosscompsL.append('seasons')
        if self.process.params.xcrossresidual:
            xcrosscompsL.append('residual')
        #Set the TS and decomposition
        if len(xcrosscompsL) == 0:
            exit('Plotting crosscorrelation requires at least one time series component to compare')
        if len(self.index) > 1 and len(xcrosscompsL) > 1:
            exit('Plotting crosscorrelation must have either a single index or a single time series component')
        multiIndex = False
        if len(self.index) > 1:
            multiIndex = True
            
        self._SetTSDecomposition()
 
        odf = {}
        
        #Decompose the indexes
        self.indexD = {}
        for item in self.index:
            if self.process.params.normalize:
                self.TS.SetTS(self._Normalize(self.index[item]))
            else:
                self.TS.SetTS(self.index[item])
            index, dummy, titleAddendum = self._tsDecompose()
            self.indexD[item] = pd.DataFrame(index, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])
            #idf[item] = pd.DataFrame(index, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])
        
        for xy in self.obs:
            if self.process.params.normalize:
                self.TS.SetTS(self._Normalize(self.obs[xy]))
            else:
                self.TS.SetTS(self.obs[xy])
            obs, dummy, titleAddendum = self._tsDecompose()

            odf[xy] = pd.DataFrame(obs, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])

        dfD = {} 
        xcrossD = {}
        #Start looping with decomposition component (= order in plot)  
        for xc in xcrosscompsL:
            dfD[xc] = {}
            xcrossD[xc] = {}
            for xy in self.obs:
                dfD[xc][xy] = {}
                xcrossD[xc][xy] = {}
                if xc == 'observed':
                    A = odf[xy].observed
                elif xc == 'seasons':
                    A = odf[xy].seasons  
                elif xc == 'residual':
                    A = odf[xy].residual
                elif xc == 'tendency':  
                    A = odf[xy].tendency
                else:
                    print ('xc',xy)
                    exit('no such decompositon component')

                Anorm = ( A - np.average(A) ) / np.std(A)
                
                for i in self.indexD:

                    self.Inorm = ( self.indexD[i][xc] - np.average(self.indexD[i][xc]) ) /  (np.std(self.indexD[i][xc]))
                    #self.Inorm = np.roll(Anorm, shift=1)
                    #print ('Anorm',Anorm)
                    #print ('Inrom',self.Inorm)
                    if self.process.params.abs:
                        lag, pearson, corrIndex, corrObs = self.TS.CrossCorrAbs(self.Inorm.values, Anorm.values,
                                self.process.params.mirrorlag,self.process.params.maxlag)
                    else:
                        lag, pearson, corrIndex, corrObs = self.TS.CrossCorr(self.Inorm.values, Anorm.values,
                                self.process.params.mirrorlag,self.process.params.maxlag)
                    xcrossD[xc][xy][i] = {'lag':lag,'pearson':pearson}
                   
                    if self.process.params.lagplot:
                        #Create the numpy matrix, datum, index and observation adjusted for the lag
                        #Rearrange the dates 
                        if lag >= 0: 
                            #Fit the seasons
                            datum = self.datumA [lag:]    
                        else:
                            lag *= -1
                            datum = self.datumA [lag:]
                        data = {'dates': datum,
                            'corrIndex': corrIndex,
                            'corrObs': corrObs,
                            }
                    else:
                        data = {'dates': self.datumA,
                            'corrIndex': self.Inorm.values,
                            'corrObs': Anorm.values,
                            }
                    dfD[xc][xy][i] = pd.DataFrame(data, columns = ['dates', 'corrIndex', 'corrObs'])
                    
                    
        self.layoutD = self.process.proc.xy.paramsD
        self.indexLayoutD = self.process.proc.index.paramsD
        graph = GraphPlot()
        if multiIndex:
            graph._GraphXCrossMultiIndex(dfD, xcrossD, self.layoutD, self.indexLayoutD, self.process.params)
        else:
            graph._GraphXCross(dfD, xcrossD, self.layoutD, self.indexLayoutD, self.process.params)
            
    def _LayerXCorrGraph(self,locus):
        '''
        '''
        if len(self.obs) > 1 and self.process.params.lagplot:
            exit('Plotting crosscorrelation with multiple samples requires that "lagplot" is set to False (default)')
        if len(self.obs) == 0 or len(self.index) == 0:
            exit('Plotting crosscorrelation requires at least one observation point and one index')
        xcrosscompsL = []
        if self.process.params.xcrossobserved:
            xcrosscompsL.append('observed')
        if self.process.params.xcrosstendency:
            xcrosscompsL.append('tendency')
        if self.process.params.xcrosseason:
            xcrosscompsL.append('seasons')
        if self.process.params.xcrossresidual:
            xcrosscompsL.append('residual')
        #Set the TS and decomposition
        if len(xcrosscompsL) == 0:
            exit('Plotting crosscorrelation requires at least one time series component to compare')
        if len(self.index) > 1 and len(xcrosscompsL) > 1:
            exit('Plotting crosscorrelation must have either a single index or a single time series component')
        multiIndex = False
        if len(self.index) > 1:
            multiIndex = True
            
        self._SetTSDecomposition()
 
        odf = {}
        
        #Decompose the indexes
        self.indexD = {}
        for item in self.index:
            if self.process.params.normalize:
                self.TS.SetTS(self._Normalize(self.index[item]))
            else:
                self.TS.SetTS(self.index[item])
            index, dummy, titleAddendum = self._tsDecompose()
            self.indexD[item] = pd.DataFrame(index, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])
            #idf[item] = pd.DataFrame(index, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])
        
        for xy in self.obs:
            if self.process.params.normalize:
                self.TS.SetTS(self._Normalize(self.obs[xy]))
            else:
                self.TS.SetTS(self.obs[xy])
            obs, dummy, titleAddendum = self._tsDecompose()

            odf[xy] = pd.DataFrame(obs, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])

        dfD = {} 
        xcrossD = {}
        #Start looping with decomposition component (= order in plot)  
        for xc in xcrosscompsL:
            dfD[xc] = {}
            xcrossD[xc] = {}
            for xy in self.obs:
                dfD[xc][xy] = {}
                xcrossD[xc][xy] = {}
                if xc == 'observed':
                    A = odf[xy].observed
                elif xc == 'seasons':
                    A = odf[xy].seasons  
                elif xc == 'residual':
                    A = odf[xy].residual
                elif xc == 'tendency':  
                    A = odf[xy].tendency
                else:
                    print ('xc',xy)
                    exit('no such decompositon component')
                Anorm = ( A - np.average(A) ) / np.std(A)
 
                for i in self.indexD:

                    self.Inorm = ( self.indexD[i][xc] - np.average(self.indexD[i][xc]) ) /  (np.std(self.indexD[i][xc]))
                    if self.process.params.abs:
                        lag, pearson, corrIndex, corrObs = self.TS.CrossCorrAbs(self.Inorm.values, Anorm.values)
                    else:
                        lag, pearson, corrIndex, corrObs = self.TS.CrossCorr(self.Inorm.values, Anorm.values)
                    xcrossD[xc][xy][i] = {'lag':lag,'pearson':pearson}
                   
                    if self.process.params.lagplot:
                        #Create the numpy matrix, datum, index and observation adjusted for the lag
                        #Rearrange the dates 
                        if lag >= 0: 
                            #Fit the seasons
                            datum = self.datumA [lag:]    
                        else:
                            lag *= -1
                            datum = self.datumA [lag:]
                        data = {'dates': datum,
                            'corrIndex': corrIndex,
                            'corrObs': corrObs,
                            }
                    else:
                        data = {'dates': self.datumA,
                            'corrIndex': self.Inorm.values,
                            'corrObs': Anorm.values,
                            }
                    dfD[xc][xy][i] = pd.DataFrame(data, columns = ['dates', 'corrIndex', 'corrObs'])
                    
                    
        self.layoutD = self.process.proc.xy.paramsD
        self.indexLayoutD = self.process.proc.index.paramsD
        graph = GraphPlot()
        if multiIndex:
            graph._GraphXCrossMultiIndex(dfD, xcrossD, self.layoutD, self.indexLayoutD, self.process.params)
        else:
            graph._GraphXCross(dfD, xcrossD, self.layoutD, self.indexLayoutD, self.process.params)

    def _compComponentGraph(self,locus):
        obs = {}
        for item in self.locusFeature[locus]:
            x,y, = item['x'],item['y']
            obs[item['id']] = self._ExtractTimeSeriesPlot(locus,x,y)
        datumA = [self.process.srcperiod.datumD[datum]['acqdate'] for datum in self.process.srcLayerD[locus]] 
        #Convert datum to numpy array and an object variabiel (object reused alter)
        self.datumA = np.array(datumA)
        return obs
            
    def _tsDecompose(self):
        '''Expects that TS is defined with a time series
        '''
        self.TS.SeasonalDecompose()
        
        if self.process.params.naive:   
            t = 'naive kernel' 
            self.process.params.trend = 'naive'
        else:
            t = self.process.params.trend

        titleAddendum = '(%(t)s, period: %(p)d filter: %(f)d [%(y)d yr])' %{'t':t,'p':self.TS.period,'f':self.TS.window,'y':self.process.params.yearfac}
        
        #Run a mann-kendall regression on the trend 
        
        regressD = {}
        ts = self.TS.tendency[~np.isnan(self.TS.tendency)]
        if not self.process.params.additive:
            #replace ts with Logarithmic of ts
            ts = np.log(ts)
            
        self.tsy = np.arange(ts.shape[0])
        mk, tsslope, tsintercept, tslowslope, tshislope = self._MKtestAlongAxis(ts)

        regressD['tendency'] = {'mk':mk,'tsslope':tsslope,'tsintercept': tsintercept}
        if self.TS.residual is None:
            pass
        else:
            ts = self.TS.residual[~np.isnan(self.TS.tendency)]
            mk, tsslope, tsintercept, tslowslope, tshislope = self._MKtestAlongAxis(ts)
            regressD['residual'] = {'mk':mk,'tsslope':tsslope,'tsintercept': tsintercept}
        data = {'dates': self.datumA,
                'observed': self.TS.ts,
                'tendency': self.TS.tendency,
                'seasons': self.TS.fullseasons,
                'residual': self.TS.residual,
                'ts': ts,
                }
        return data, regressD, titleAddendum
                        
    def _ComponentGraph(self,dataD):
        '''
        '''
        #Set the TS and decomposition
        self._SetTSDecomposition()
        
        df = {} 
        regressD = {}
        for item in dataD:
            if self.process.params.normalize:
                self.TS.SetTS(self._Normalize(dataD[item]))
            else:
                self.TS.SetTS(dataD[item])
            data, regressD[item], titleAddendum = self._tsDecompose()

            df[item] = pd.DataFrame(data, columns = ['dates', 'observed', 'tendency', 'seasons', 'residual', 'ts'])
            
        self.process.params.title = '%(t)s: %(ta)s' %{'t':self.process.params.title,'ta':titleAddendum}
        
        graph = GraphPlot()
        columns = ['observed', 'tendency', 'seasons', 'residual']
        graph._GraphDecomposedTrend(df, columns, regressD, self.layoutD, self.process.params)
   
    def _MKtestAlongAxis(self,ts):
        #from timeseries.timeseries
        from geoimagine.ktnumba import MKtestIni
        from scipy.stats import mstats

        if np.isnan(np.sum(ts)):
            return False

        mk = MKtestIni(ts)
        tsslope, tsintercept, tslowslope, tshislope = mstats.theilslopes(ts,self.tsy)

        return mk,tsslope, tsintercept, tslowslope, tshislope 

    def _OLSAlongAxis(self,ts):
        #from timeseries.timeseries
        from geoimagine.ktnumba import OLSextendedNumba
        self.counter += 1
        if np.isnan(np.sum(ts)):
            self.olsA[:,self.counter] = self.olsNullA
            return 0
        self.olsA[:,self.counter] =  OLSextendedNumba(self.tsy, ts, self.olsArr)
        return 0
        
    def _ExtractTimeSeriesPlot(self,locus,x,y):
        '''
        '''
        bbox = [ x, x, y, y ]
        valL =[]  
        for datum in self.process.srcLayerD[locus]:
            if not self.process.srcLayerD[locus][datum]:
                continue
            for comp in self.process.srcLayerD[locus][datum]:
                if not self.process.srcLayerD[locus][datum][comp]:
                    continue
                aVal,src_offset = extract.ExtractRasterLayer(self.process.srcLayerD[locus][datum][comp].layer, bbox)
                valL.append(aVal[0][0])
        return np.array( valL )
        
