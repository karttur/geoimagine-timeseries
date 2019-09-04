'''
Created on 19 Oct 2018

@author: thomasgumbricht
'''
import geoimagine.gis.mj_gis_v80 as mj_gis
from geoimagine.support import karttur_dt as mj_dt
from geoimagine.timeseries import seasonalts as mj_ts
from os import path
import numpy as np
from pandas import to_datetime, Series
from math import floor

class TScoreFuncs:
    '''Core timeseries functions
    '''
    def __init__(self):
        '''Empty call to access the functions
        '''
        pass
    
    def _SetMinMax(self,a):
        '''Numpy array min, max and range extraction
        '''
        self.min = np.amin(a)
        self.max = np.amax(a)
        self.range = self.max-self.min
        
    def _Normalize(self,a):
        '''Numpy array Normalization
        '''
        self._SetMinMax(a)
        return (a - self.min)/self.range
        
        '''
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(a)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
        return scaler.transform(a)
        '''
        '''
        norm = np.linalg.norm(a)
        if norm == 0: 
            return a
        return a / norm
        '''
    
    def _Standardize(self,a):
        '''Numpy array Standardization
        '''
        from sklearn.preprocessing import StandardScaler
        #from math import sqrt
        # load the dataset and print the first 5 rows
        #series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
        #print(series.head())
        # prepare data for standardization
        #values = series.values
        #values = values.reshape((len(values), 1))
        # train the standardization
        scaler = StandardScaler()
        scaler = scaler.fit(a)
        #print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
        # standardization the dataset and print the first 5 rows
        return scaler.transform(a)
        '''
        for i in range(5):
            print(normalized[i])
        # inverse transform and print the first 5 rows
        inversed = scaler.inverse_transform(normalized)
        for i in range(5):
            print(inversed[i])
        '''
    
    def _FillAlongAxis(self,ts):
        '''Linear interpolation of NaN, calls Numba function
        '''
        from geoimagine.ktnumba import InterpolateLinearNaNNumba
        if np.all(np.isnan(ts)):
            return self.dstNullDarr[self.dstcomp]
        if np.isnan(np.sum(ts)):
            non_nans = (~np.isnan(ts)).sum()
            if float(non_nans)/ts.shape[0] < self.process.params.validfraction:
                return self.dstNullDarr[self.dstcomp]
            avg = np.nanmean(ts)
            if np.isnan(ts[0]):
                ts[0] = avg
            if np.isnan(ts[ts.shape[0]-1]):
                ts[ts.shape[0]-1] = avg
            ts = InterpolateLinearNaNNumba(ts)
            return ts
        else:
            return ts
        
    def _ExtractSeason(self,ts):
        '''Extract seasonal signal, calls Numba function
        '''
        from geoimagine.ktnumba import ResampleSeasonalAvg
        if np.all(np.isnan(ts)):
            return self.dstNullDarr[self.dstcomp] 
        tsr = ResampleSeasonalAvg(ts, self.dstArr, self.seasons)
        return self._FillAlongAxis(tsr)

    def _SeasonFill(self,ts,season):
        '''Fill timeseries null with seasonal interpolation, calls Numba function
        '''
        from geoimagine.ktnumba import SeasonFill
        if np.all(np.isnan(season)):
            return self.dstNullDarr[self.dstcomp]
        if np.isnan(ts[0]):
            ts[0] = season[0]
        if np.isnan(ts[self.lastitem]):
            ts[self.lastitem] = season[self.lastitem]

        tsr = SeasonFill(ts,season)
        return tsr
    
    def _OverAllAvg(self,ts):
        '''Time series overall average
        '''
        if np.isnan(np.sum(ts)):
            return self.dstNullD['avg']
        return np.average(ts)
    
    def _OverAllStd(self,ts):
        '''Time series overall standard deviation
        '''
        if np.isnan(np.sum(ts)):
            return self.dstNullD['std']
        return np.std(ts)
    
    def _OverAllNanMean(self,ts):
        '''Time series overall average
        '''
        if np.isnan(np.nanmean(ts)):
            return self.dstNullD['mean']
        if self.kernelflag:
            if self.process.params.wrap:       
                ts = np.lib.pad(ts, (self.halfkernel, self.halfkernel), 'wrap')
                ts = np.convolve(ts, self.kernel, mode='valid')
            else:
                ts = np.convolve(ts, self.kernel, mode='same')
        if self.threshold:
            ts[ts > self.maxthreshold] = np.nan
            ts[ts < self.minthreshold] = np.nan
        return np.nanmean(ts)
    
    def _OverAllNanStd(self,ts):
        '''Time series overall standard deviation
        '''
        if np.isnan(np.nanmean(ts)):
            return self.dstNullD['std']
        if self.kernelflag:
            if self.process.params.wrap:
                ts = np.lib.pad(ts, (self.halfkernel, self.halfkernel), 'wrap')
                ts = np.convolve(ts, self.kernel, mode='valid')
            else:
                ts = np.convolve(ts, self.process.params.filter, mode='same')
        if self.threshold:
            ts[ts > self.maxthreshold] = np.nan
            ts[ts < self.minthreshold] = np.nan
        return np.nanstd(ts)
    
    def _MKtestAlongAxis(self,ts):
        '''Mann Kendall and Theil Sen statistics for time series
        '''
        from geoimagine.ktnumba import MKtestIni
        from scipy.stats import mstats
        self.counter+=1
        if np.isnan(np.sum(ts)):
            self.mktsA[:,self.counter] = self.mktsNullA
            return 0
        self.mktsA[0,self.counter] = MKtestIni(ts)
 
        self.mktsA[1:,self.counter] = mstats.theilslopes(ts,self.tsy)

        return 0

    def _OLSAlongAxis(self,ts):
        '''Ordinary Least Square statistics for time series
        '''
        from geoimagine.ktnumba import OLSextendedNumba
        self.counter += 1
        if np.isnan(np.sum(ts)):
            self.olsA[:,self.counter] = self.olsNullA
            return 0
        self.olsA[:,self.counter] =  OLSextendedNumba(self.tsy, ts, self.olsArr)
        return 0
    
    def _SeasonalAutoCorrFIX(self,ts,partial):
        '''
        from mj_pandas_v67 import PandasTimeSeries,Autocorrelate
        
        if ts[0] == -32768:
            for l in range(13):
                self.lagL[l].append(-32768)
            return -32768  
        '''
        if np.isnan(np.sum(ts)):
            return self.dstNullDarr[self.dstcomp]
        self.pandasTS = PandasTimeSeries(self.process.period)
        self.pandasTS.CreatePandasTS(np.array(ts))
        tsm = self.ResampleToMonthAvg()
        lagAA = np.zeros([ 12, 13])
        for m in range(0,12):        
            sig = np.roll(tsm.values, shift=m)
            lagA = Autocorrelate(sig,11,partial)
            lagB = Autocorrelate(np.flipud(sig),11,partial)
            lagAA[m:] = np.concatenate((lagA[0:7],np.flipud(lagB[0:6])))
        monthAvgAutoCorr = np.apply_along_axis( self.AxisAverage, 0, lagAA )
        for l in range(13):
            self.lagL[l].append(monthAvgAutoCorr[l])
        return -32768
    
    def _FullAutoCorr(self,ts,partial):
        '''Autocorrelation of time series
        '''
        #from statsmodels.tsa.stattools import acf, pacf
        from statsmodels import api as sm
        if np.isnan(np.sum(ts)):
            #return self.dstNullDarr[self.dstcomp]
            self.lagL = self.dstNullDarr[self.dstcomp]
            return -32768
        if partial:
            #I had to convert the numpy array to list, otherwise it did not work
            lagA = sm.tsa.pacf(ts.tolist(),nlags=self.process.params.nlags)
        else:
            #I had to convert the numpy array to list, otherwise it did not work
            lagA = sm.tsa.acf(ts.tolist(),nlags=self.process.params.nlags)

        #concat and flip removed, already done above in this version
        #lagAA = np.concatenate((lagA[0:7],np.flipud(lagA[0:6])))

        for l in range(self.process.params.nlags):
            self.lagL[l].append(lagA[l])
        return -32768
    
    def _IndexCrossTrendsAlongAxis(self,ts):
        '''Cross Correlation of index and image (pixelwise) timeseries
        '''
        if np.isnan(np.sum(ts)):
            #Set all output arrays to NaN
            self.obs.append(self.tsNullArray)
            self.tendency.append(self.tsNullArray)
            self.fullseasons.append(self.tsNullArray)
            self.residual.append(self.tsNullArray)
            #self.seasons.append(self.tsNullArray)
            #Return a dummy
            return -32768
        
        self.obs.append(ts)
        
        #Set the timeseries
        self.TS.SetTS(ts)
        #run the decomposition
        if self.dodecompose:
            self.TS.SeasonalDecompose()
    
            #append the results to the arrays
            if self.process.params.xcrosstendency:
                self.tendency.append(self.TS.tendency)
            if self.process.params.xcrosseason:
                self.fullseasons.append(self.TS.fullseasons)
            if self.process.params.xcrossresidual:
                self.residual.append(self.TS.residual)
        
        #Return a dummy
        return 0
        
    def _LayerXCorrSetTStoNull(self):
        '''Set pixelwise timeseries to NaN (for layer cross correlation)
        '''
        #Set all output arrays to NaN
        if self.process.params.xcrossobserved:
            self.obs_master.append(self.tsNullArray)
            self.obs_slave.append(self.tsNullArray)
        if self.process.params.xcrosstendency:
            self.tendency_master.append(self.tsNullArray)
            self.tendency_slave.append(self.tsNullArray)
        if self.process.params.xcrosseason:
            self.fullseasons_master.append(self.tsNullArray)
            self.fullseasons_slave.append(self.tsNullArray)
        if self.process.params.xcrossresidual:
            self.residual_master.append(self.tsNullArray)
            self.residual_slave.append(self.tsNullArray)
        
        #self.seasons_master.append(self.tsNullArray)
        #self.seasons_slave.append(self.tsNullArray)

    def _LayerXCorrDecomposeTS(self,master,slave):
        '''Decompose pixelwise timeseries (for layer cross correlation)
        '''
        #Set the observed time series
        self.obs_master.append(master)
        self.obs_slave.append(slave)
        if self.dodecompose:
            
            #Set the timeseries
            self.TS.SetTS(master)
            
            #run the appropriate analysis
            self.TS.SeasonalDecompose()
            #append the results to the arrays
            #self.obs_master.append(self.TS.ts)
            if self.process.params.xcrosstendency:
                self.tendency_master.append(self.TS.tendency)
            if self.process.params.xcrosseason:
                self.fullseasons_master.append(self.TS.fullseasons)
            if self.process.params.xcrossresidual:
                self.residual_master.append(self.TS.residual)
            #self.seasons_master.append(self.TS.seasons)
            
            #Set the timeseries
            self.TS.SetTS(slave)
            #run the decomposition
            self.TS.SeasonalDecompose()
            #append the results to the arrays
            #self.obs_slave.append(self.TS.ts)
            if self.process.params.xcrosstendency:
                self.tendency_slave.append(self.TS.tendency)
            if self.process.params.xcrosseason:
                self.fullseasons_slave.append(self.TS.fullseasons)
            if self.process.params.xcrossresidual:
                self.residual_slave.append(self.TS.residual)
                                            
        #Return dummy
        return 1
    
    def _IndexXCorr(self,A):
        '''Cross correlate index and pixelwise timeseries
        '''
        #if only null, then lag and pearson = null
        if np.isnan(np.sum(A)):
            self.xcross['lag'].append(-32768)
            self.xcross['pearson'].append(-32768)
            return
        if self.process.params.normalize:  
            Anorm = ( A - np.average(A) ) /  (np.std(A))
        else:
            Anorm = A
        if self.process.params.abs:
            lag, pearson, corrIndex, corrObs = self.TS.CrossCorrAbs(self.Inorm, Anorm,
                    self.process.params.mirrorlag,self.process.params.maxlag)
        else:
            lag, pearson, corrIndex, corrObs = self.TS.CrossCorr(self.Inorm, Anorm,
                    self.process.params.mirrorlag,self.process.params.maxlag) 
                                                                
        #Append the results one by one
        self.xcross['lag'].append(lag)
        self.xcross['pearson'].append(pearson)
        
    def _LayerXCorr(self, mArr, sArr):
        '''Cross correlate two pixelwise timeseries
        '''
        #if only null, then lag and pearson = null
        if np.isnan(np.sum(mArr)):
            self.xcross['lag'].append(-32768)
            self.xcross['pearson'].append(-32768)
            return
        if self.process.params.normalize:
            mArr = ( mArr - np.average(mArr) ) /  (np.std(mArr))
            sArr = ( sArr - np.average(sArr) ) /  (np.std(sArr))
            '''
            print ('norm mArr',mArr)
            print ('norm sArr',sArr)
            '''
        if self.process.params.abs:
            lag, pearson, corrIndex, corrObs = self.TS.CrossCorrAbs(mArr, sArr, self.process.params.mirrorlag, self.process.params.maxlag)
            
        else:
            lag, pearson, corrIndex, corrObs = self.TS.CrossCorr(mArr, sArr,self.process.params.mirrorlag,self.process.params.maxlag)

        #Append the results one by one
        self.xcross['lag'].append(lag)
        self.xcross['pearson'].append(pearson)
        
    def _LayerCrossCorrFixedLag(self, mArr, sArr, lag):
        '''Cross correlate timeseries at fixed lag
        '''
        #if only null, then pearson = null = -32768
        if np.isnan(np.sum(mArr)):
            self.xcorr.append(-32768)
            return 
        if self.process.params.normalize:
            mArr = ( mArr - np.average(mArr) ) /  (np.std(mArr))
            sArr = ( sArr - np.average(sArr) ) /  (np.std(sArr))

        pearson = self.TS._CrossCorrFixedLag(mArr, sArr, lag)
        self.xcorr.append(pearson)
                
    def _SetTSDecomposition(self):
        if self.process.params.naive:
            '''
            if self.process.params.additive:
                model = 'additive'
            else:
                model = 'multiplicative'
            
            self.TS = mj_ts.NaiveTS(self.process.srcperiod, self.process.params.yearfac, 
                    self.process.params.trend, self.process.params.kernel, model) 
            '''
            self.TS = mj_ts.NaiveTS(self.process.srcperiod, self.process.params)    
        else:
            '''
            self.TS = mj_ts.SeasonalTS(self.process.srcperiod, 
                    self.process.params.yearfac, self.process.params.trend, self.process.params.kernel, 
                    self.process.params.prefilterseason, self.process.params.forceseason)
            '''
            self.TS = mj_ts.SeasonalTS(self.process.srcperiod, self.process.params)
            
    def _IndexDecomposition(self):
        '''Decompose index timeseries
        '''             
        self.indexD = {}
        indexL = list(self.process.proc.index.paramsD.keys())
        for i in indexL:
            self.indexD[i] = {}
            ts = np.array( [x[1] for x in self.session._SelectClimateIndex(self.process.srcperiod,i)] )
            
            #set the timesereis
            self.TS.SetTS(ts)

            #Run the decomposition
            self.TS.SeasonalDecompose()
            
            #Extract the data
            self.indexD[i]['obs'] = self.TS.ts[:]
            self.indexD[i]['tendency'] = self.TS.tendency[:]  
            self.indexD[i]['residual'] = self.TS.residual[:] 
            #self.indexD[i]['seasons'] = self.TS.seasons[:]
            self.indexD[i]['fullseasons'] = self.TS.fullseasons[:]
            self.indexD[i]['season'] = self.TS.fullseasons[:]

class TScommon(TScoreFuncs):
    '''common class for timeseries and timeseriesgraph
    '''    
    def __init__(self, process, session, verbose):
        TScoreFuncs.__init__(self)
        self.verbose = verbose
        self.process = process   
        self.session = session
        
    def _CheckSrcTS(self):
        '''Check source timeseries consistency
        '''
        srccompL = []

        for locus in self.process.srcLayerD:
            if len(self.process.srcLayerD[locus]) == 0:
                exitstr = 'EXITING, no dates defined in timeseries'
                exit(exitstr)
            for datum in self.process.srcLayerD[locus]:
                if not self.process.srcLayerD[locus][datum]:
                    print ('    missing datum', datum)
                    SNULLEBULLE
                    continue
                elif len(self.process.srcLayerD[locus][datum]) == 0:
                    exitstr = 'EXITING, no compositions defined in timeseries'
                    exit(exitstr)
                for comp in self.process.srcLayerD[locus][datum]:
                    if not comp in srccompL:
                        srccompL.append(comp)
        return srccompL
        
    def _CheckSrcDstTS(self):
        '''Check timeseries consistency
        '''            
        srccompL = self._CheckSrcTS()
        dstcompL = []
        for locus in self.process.dstLayerD:
            if len(self.process.dstLayerD[locus]) == 0:
                exitstr = 'EXITING, no dates defined in timeseries'
                print (exitstr)
                SNULLEBULLE
                exit(exitstr)
            for datum in self.process.dstLayerD[locus]:
                if not self.process.dstLayerD[locus][datum]:
                    print ('    missing datum', locus, datum, self.process.dstLayerD[locus][datum])
                    print ('datum keys', list(self.process.dstLayerD[locus][datum].keys()))
                    SNULLEBULLE
                    continue
                elif len(self.process.dstLayerD[locus][datum]) == 0:
                    exitstr = 'EXITING, no compositions defined in Sentinel._ExtractSentinelTileCoords'
                    exit(exitstr)
                for comp in self.process.dstLayerD[locus][datum]:
                    if not comp in dstcompL:
                        dstcompL.append(comp)
        return srccompL, dstcompL
                
    def _OpenSrcLayers(self,locus,comp):
        '''Open timeseries source datalayers for reading
        '''
        geoFormatD = {}
        self.OpenLayers = []
        self.LayerForAcqDateStrD = {}
        self.AcqDateStrSrcLayerD ={}
        firstLayer = True
        self.srcDateD[comp] = []
        for datum in self.process.srcLayerD[locus]:
            if not self.process.srcLayerD[locus][datum]:
                print ('    missing datum', datum)
                continue
            if not self.process.srcLayerD[locus][datum][comp]:
                print ('    missing comp datum', comp, datum)
                continue
            self.srcDateD[comp].append(datum)
            if self.process.srcLayerD[locus][datum][comp].FPN in self.OpenLayers:
                #Duplicate layer reading, for seasonal or static data combined with timesereis
                #print (self.process.srcLayerD[locus][datum][comp].datum.acqdatestr)
                self.process.srcLayerD[locus][datum][comp].copy = True
                self.AcqDateStrSrcLayerD[datum] = self.process.srcLayerD[locus][datum][comp].datum.acqdatestr
            
            else:
                #print ('opening',locus,datum,comp,self.process.srcLayerD[locus][datum][comp].FPN)

                self.process.srcLayerD[locus][datum][comp].copy = False
                self.process.srcLayerD[locus][datum][comp].srcDS,self.process.srcLayerD[locus][datum][comp].layer = mj_gis.RasterOpenGetFirstLayer(self.process.srcLayerD[locus][datum][comp].FPN,'read')
                self.OpenLayers.append(self.process.srcLayerD[locus][datum][comp].FPN)
                self.LayerForAcqDateStrD[self.process.srcLayerD[locus][datum][comp].datum.acqdatestr] = datum
                
                self.process.srcLayerD[locus][datum][comp].layer.GetGeometry()
                #print (self.process.srcLayerD[locus][datum][comp].layer.lins)
                #print (self.process.srcLayerD[locus][datum][comp].layer.cols)
                lins = self.process.srcLayerD[locus][datum][comp].layer.lins
                cols = self.process.srcLayerD[locus][datum][comp].layer.cols
                if comp in self.LayerReaderD:
                    if lins != self.LayerReaderD[comp]['lins']:
                        ERROR
                    if cols != self.LayerReaderD[comp]['cols']:
                        ERROR
                else:
                    self.LayerReaderD[comp] = {'lins':lins,'cols':cols}
                
                l = self.process.srcLayerD[locus][datum][comp].layer
                self.srcCellNullD[comp] = self.process.srcLayerD[locus][datum][comp].layer.cellnull
                if hasattr(self.process.srcLayerD[locus][datum][comp].comp,'id'):
                    self.idD[self.process.srcLayerD[locus][datum][comp].comp.id] = comp
                if firstLayer: 
                    self.geoFormatD[comp] = {'lins':l.lins,'cols':l.cols,'projection':l.projection,'geotrans':l.geotrans,'cellsize':l.cellsize}
                    geoFormatD[locus] = {'lins':l.lins,'cols':l.cols,'projection':l.projection,'geotrans':l.geotrans,'cellsize':l.cellsize}
                    self.firstFPN = self.process.srcLayerD[locus][datum][comp].FPN
                    firstLayer = False
                else:
                    gfD = {'lins':l.lins,'cols':l.cols,'projection':l.projection,'geotrans':l.geotrans,'cellsize':l.cellsize}
                    for item in geoFormatD[locus]:
                        if geoFormatD[locus][item] != gfD[item]:
                            if item == 'cellsize':
                                if round(geoFormatD[locus][item],4) != round(gfD[item],4):
                                    print ('layers can not be processed together (%s = %s and %s = %s' %(item, geoFormatD[locus][item], item, gfD[item]))
                                    STOPPAPRESSARNA
                            elif item == 'geotrans':
                                pass
                            else:
                                print ('layers can not be processed together (%s = %s and %s = %s' %(item, geoFormatD[locus][item], item, gfD[item]))
                                print ('layer 0 ', self.firstFPN)
                                print ('layer 1', self.process.srcLayerD[locus][datum][comp].FPN)
                                print (geoFormatD[locus][item])
                                print (gfD[item])
                                STOPPAPRESSARNA
          
    def _CloseSrcRasterFiles(self, locus):
        '''Close open raster files
        '''
        for datum in self.process.srcLayerD[locus]:
            if not self.process.srcLayerD[locus][datum]:
                continue
            for comp in self.process.srcLayerD[locus][datum]:
                if not self.process.srcLayerD[locus][datum][comp]:
                    continue
                #print ('closing',locus,datum,comp,self.process.srcLayerD[locus][datum][comp].FPN)
                self.process.srcLayerD[locus][datum][comp] = None
                            
        for datum in self.process.dstLayerD[locus]:
            for comp in self.process.dstLayerD[locus][datum]:
                if self.process.dstLayerD[locus][datum][comp]:
                    self.process.dstLayerD[locus][datum][comp].dstDS.CloseDS()
                    self.session._InsertLayer(self.process.dstLayerD[locus][datum][comp],self.process.overwrite,self.process.delete)

    def _ResampleToAnnualAvg(self,ts):
        '''
        '''
        from geoimagine.ktnumba import ResampleToPeriodAvg
        if np.isnan(np.sum(ts)):
            return self.dstNullDarr[self.dstcomp]
        tsr = ResampleToPeriodAvg(ts, self.dstArr, self.resampleperiods, self.years)
        return tsr
    
    def _ResampleToAnnualSum(self,ts):
        '''
        '''
        from geoimagine.ktnumba import ResampleToPeriodSum
        if np.isnan(np.sum(ts)):
            return self.dstNullDarr[self.dstcomp]
        tsr = ResampleToPeriodSum(ts, self.dstArr, self.resampleperiods, self.years)
        return tsr
    
    def _ResampleDictPeriodAvg(self,ts):
        '''
        '''
        from geoimagine.ktnumba import ResampleToDictPeriodAvgNan
        if np.all(np.isnan(ts)):
            return self.dstNullDarr[self.dstcomp]
        tsr = ResampleToDictPeriodAvgNan(ts, self.dstArr, self.numbaCols)
        #Set nan to dstnull
        tsr[np.isnan(tsr)] = self.dstNull
        return tsr
    
    def _ResampleInterpolPeriodAvg(self,ts):
        '''
        '''
        from geoimagine.ktnumba import InterpolateLinearFixedNaNNumba, ResampleToDictPeriodAvgNan
        if np.all(np.isnan(ts)):
            return self.dstNullDarr[self.dstcomp]
        tsi = InterpolateLinearFixedNaNNumba(ts,self.AcqSteps,self.interpolSteps,self.interpolA) 
        tsr = ResampleToDictPeriodAvgNan(tsi, self.dstArr, self.numbaCols)
        #tsr[np.isnan(tsr)] = self.dstNull
        return tsr
    
class ProcessTimeSeries(TScommon):
    '''class for time series processing
    '''   
    def __init__(self, process, session, verbose):
        TScommon.__init__(self, process, session, verbose)
        self.verbose = True
        if self.verbose:
            print ('    starting ProcessTimeSeries:', self.process.proc.processid)
        self._maskLayer = False
        #Organize the timeseries data
        srccompL, dstcompL = self._CheckSrcDstTS()  
        #Dict for geoformats 
        self.geoFormatD = {}
        #Dict for source data null
        self.srcCellNullD = {}

        #The time series processing is run location by location
        for locus in self.process.srcLayerD:
            print ('        locus',locus)
            self.locus = locus
            #Check that all the dst files are already done and not to be overwritten
            self._CheckDstRasterFiles()
            if self.SkipLocus:
                if self.verbose:
                    print ('            Locus all finished, continuing')
                continue
            #Crate a dict of the datum src and dst layers to create
            self.dstDateD = {}
            self.srcDateD = {}
            for srccomp in srccompL:
                self.srcDateD[srccomp] = []
            for dstcomp in dstcompL:
                self.dstDateD[dstcomp] = []

            #dictionary linking comp to id
            self.idD = {} 
            self.LayerReaderD = {}
            
            #open the input data
            for comp in srccompL:
                self._OpenSrcLayers(locus,comp)
                
        
            #Loop over the src data to find valid date list (if none, then return)
            for sc in self.srcDateD:
                if self.srcDateD[sc]:
                    self.srccomp = sc
            self.srcDateL = self.srcDateD[self.srccomp]
                    
            #Adjust the output related to the input            
            if self.process.proc.processid.lower()[0:10] == 'resamplets':
                firstSrcDate = self.process.srcperiod.datumD[self.srcDateL[0]]['acqdate']
                lastSrcDate = self.process.srcperiod.datumD[self.srcDateL[-1]]['acqdate']
                if self.process.srcperiod.timestep in ['M','monthlyday']:
                    lastSrcDate = mj_dt.AddMonth(lastSrcDate, 1)

                datumL = list(self.process.dstLayerD[locus].keys())

                acqdateL = [self.process.dstperiod.datumD[item]['acqdate'] for item in datumL]
                #self._maskLayer = True
                #self._maskLayerD = {'datum':self.srcDateL[0],'comp':srccompL[0]}

                #Loop all datums in output and make sure all the required input is available 
                okDateL = []
                notOkDateL = []
                if self.process.dstperiod.timestep[len(self.process.dstperiod.timestep)-1] =='D':
                    for x,datum in enumerate(acqdateL):
                        startdate = datum
                        enddate = mj_dt.DeltaTime(startdate, self.process.dstperiod.periodstep-1)
                        if startdate >= firstSrcDate and enddate <= lastSrcDate:
                            okDateL.append(datumL[x])
                        else:
                            notOkDateL.append(datumL[x])
                elif self.process.dstperiod.timestep == 'M':
                    for x,datum in enumerate(acqdateL):
                        startdate = datum
                        enddate = mj_dt.AddMonth(startdate, 1)
                        enddate = mj_dt.DeltaTime(enddate , -1)
                        if startdate >= firstSrcDate and enddate <= lastSrcDate:
                            okDateL.append(datumL[x])       
                        else:
                            print ('adding not ok', datumL[x],startdate, firstSrcDate, enddate, lastSrcDate)
                            notOkDateL.append(datumL[x])
                            if self.process.proc.acceptmissing:
                                firstSrcDate = startdate
                                lastSrcDate = enddate
                                
                elif self.process.dstperiod.timestep == 'A':
                    for x,datum in enumerate(acqdateL):
                        startdate = datum
                        enddate = mj_dt.AddYear(startdate, 1)
                        enddate = mj_dt.DeltaTime(enddate , -1)
                        if startdate >= firstSrcDate and enddate <= lastSrcDate:
                            okDateL.append(datumL[x])       
                        else:
                            if self.process.proc.acceptmissing:
                                firstSrcDate = startdate
                                lastSrcDate = enddate
                            else:
                                print (startdate, firstSrcDate)
                                print (enddate, lastSrcDate)
                                print (datumL[x])
                                notOkDateL.append(datumL[x])
                                if self.process.proc.acceptmissing:
                                    firstSrcDate = startdate
                                    lastSrcDate = enddate

                else:
                    exit('Unknown timestep in PrcessTimeSeries',self.process.dstperiod.timestep)

                if self.verbose:
                    print ('notOkDateL',notOkDateL)
                print ('dsttadums', self.process.dstLayerD[locus])
                #Remove the destination dates that are not OK unless acceptmissing is True
                if not self.process.proc.acceptmissing:
                    for item in notOkDateL:
                        self.process.dstLayerD[locus].pop(item)
                        
            if self.process.proc.processid.lower()[0:7] == 'trendts':
                #The null have to be reset after the original data for trendts
                cellNullEditL = ['avg','std','ts-ic','ols-ic','ols-rmse']
                srccomp = srccompL[0]
                self.dstcompL = dstcompL
                self.dstNullD = {}
                for citem in dstcompL:
                    if citem in cellNullEditL:
                        self.process.proc.dstcompD[citem]['cellnull'] = self.srcCellNullD[srccomp]
                        #reset comp for all layers
                        for datum in self.process.dstLayerD[locus]:

                            self.process.dstLayerD[locus][datum][citem].comp.cellnull = self.srcCellNullD[srccomp]

                    self.dstNullD[citem] = self.process.proc.dstcompD[citem]['cellnull']

                self.tsy = np.arange(len(self.process.srcperiod.datumL))
    
                if 'mk' in dstcompL:
                    self.mktsNullA = np.array([self.dstNullD['mk'],self.dstNullD['ts-mdsl'],self.dstNullD['ts-ic'],self.dstNullD['ts-losl'],self.dstNullD['ts-hisl']])
                    self.mktsItemD = {'mk':0,'ts-mdsl':1,'ts-ic':2,'ts-losl':3,'ts-hisl':4}
                if 'ols' in dstcompL:
                    self.olsNullA = np.array( [self.dstNullD['ols'],self.dstNullD['ols-ic'],self.dstNullD['ols-r2'],self.dstNullD['ols-rmse']])
                    self.olsArr = np.zeros( ( 4 ), np.float32)
                    self.olsItemD = {'ols':0,'ols-ic':1,'ols-r2':2,'ols-rmse':3}
                        
            #Get the dimensions and reading/writing style for this locus
            self._SetReadWriteDim(locus)       


            #ResampleTS need to checkthe dts layers to produce prior to defining them
            if self.process.proc.processid.lower()[0:10] ==  'resamplets' and self.process.dstperiod.timestep[len(self.process.dstperiod.timestep)-1] in ['D','M']:
                self._SetNumbaArrayReader(locus)
            #Create the output data
            self._CreateOpenDstRasterFiles(locus)
            
            
            #Loop over the dst data to find out the dates to process, if all dst are False, then nothing to process
            self.dstcomp = False
            for dc in self.dstDateD:
                print ('dc',dc)
                print (self.dstDateD[dc])
                if self.dstDateD[dc][0]:
                    self.dstcomp = dc
            if not self.dstcomp:
                if self.verbose:
                    print ('Nothing to process in PrcessTimeSeries, continue to next locus')           
                continue

            self.dstDateL = self.dstDateD[self.dstcomp]
            self.dstDatum = self.dstDateL[0]

            #Set destination nullarrays
            self._CreateDstNullArrays(dstcompL)
            
            #Identify particular processes that require special settings
            
                
            if 'xcross' in self.process.proc.paramsD and self.process.proc.paramsD['xcross']:
                if self.process.proc.processid.lower()[0:5] == 'index':
                    self._SetTSDecomposition()
                    self._IndexDecomposition()
                else:
                    self._SetTSDecomposition()
                self.tsNullArray = np.ones(len(self.srcDateL), np.float32)
                self.tsNullArray[self.tsNullArray==1] = np.nan
                
            if self.process.proc.processid.lower()[0:15] == 'setassimilation':
                self.dstNullD = {}
                for item in dstcompL:
                    self.dstNullD[item] = self.process.proc.dstcompD[item]['cellnull']
                self.dstNullD['mean'] = self.dstNullD['std'] = self.dstNullD['slvavg']
                
            if hasattr(self.process.params, 'kernel'):
                if self.process.params.kernel == '0':
                    self.kernel = 0  
                    self.kernelflag = False
                elif len(self.process.params.kernel) > 5 and ',' in self.process.params.kernel:
                    self.kernelflag = True
                    self.kernel = np.array([float(x) for x in self.process.params.kernel.split(',')])
                    self.kernel /= self.kernel.sum()
                    self.halfkernel = int(floor(self.kernel.shape[0]/2)) 
                else:
                    exitstr = 'Something wrong with the kernel' 
                    exit(exitstr)
                    
            if hasattr(self.process.params, 'maxthreshold'):
                if self.process.params.maxthreshold <= self.process.params.minthreshold:
                    self.threshold = False
                else:
                    self.maxthreshold = self.process.params.maxthreshold
                    self.minthreshold = self.process.params.minthreshold
                    self.threshold = True
                    
                   
                
            #Set the nr of the last item in each ts
            self.lastitem = len(self.process.srcperiod.datumL)-1
            
            if self._LoopMaskLayer(locus):
                self.ProcessLoop(locus, srccompL, dstcompL)
            self._CloseDstRasterFiles(locus)
            self._CloseSrcRasterFiles(locus)
        
    def _LoopMaskLayer(self,locus):
        if not self._maskLayer:
            return True
            
    def ProcessLoop(self, locus, srccompL, dstcompL):
        #Process the timeseries block by block
        for l in range(self.readitems):
            self.wl = l*self.blocklines #self.wl for read and writeline in order for gdal to get info on where in the file to qrite
            '''
            if l > 3:
                continue
            '''
            print ('        line', l, self.wl, self.readitems)
    
            #Read the data
            self.srcRD = {}
            for comp in srccompL:
                print ('self.lastFullReadItem',self.lastFullReadItem)
                
                blockSize = self.LayerReaderD[comp]['blocksize']
                blockLines = self.LayerReaderD[comp]['blocklines']
                #Check the reading to not go beyond the image 
                if l == self.lastFullReadItem:
                    #SNULLE
                    #blockLines = self.lastBlockLines
                    #blockSize = self.lastBlockSize
                    blockLines = self.LayerReaderD[comp]['lastBlockLines']
                    
                    blockSize = self.LayerReaderD[comp]['lastBlockSize']
                    #print ('last lines',blockSize)
                #print ('blocksize',blockSize)  
                #print ('blocklines',blockLines)  
                #Also reset the gobal bolcklines for writing
                self.blocklines = blockLines
                self._ReadSrcBlock(locus,comp,blockSize,blockLines)
    
                #Set null to nan for the processing
                cellnull = self.srcCellNullD[comp]
                self.srcRD[comp][self.srcRD[comp]==cellnull]=np.nan
    
                if 'spl3smp' in self.process.proc.srccompD[comp]['product'].lower():
                    if self.process.proc.processid.lower() not in ['seasonfilltsmodissingletile','seasonfilltssmap']:
                        #soil moisture = 0.02 is a default fallback in SMAP that is useless, better set to null in the resample
                        self.srcRD[comp][self.srcRD[comp] <= 0.02]=np.nan
    
            #direct to process
            if self.process.proc.processid.lower() == 'mendancillarytimeseries':
                self.dstDateL = self.dstDateD[self.dstcomp]
                self._TSFillNoData()
                
            elif self.process.proc.processid.lower() == 'average3ancillarytimeseries':
                self.srccompL = srccompL
                self.dstDateL = self.dstDateD[self.dstcomp]
                self.dstNull = self.process.proc.dstcompD[dstcompL[0]]['cellnull']
                self._AverageTSPerDate()
                
            elif self.process.proc.processid[0:13].lower() == 'extractseason':
                self.dstNull = self.process.proc.dstcompD[dstcompL[0]]['cellnull']
                self.dstArr = np.ones( ( len(self.dstDateL) ), np.float32)
                self.seasons = len(self.dstDateL)
                self._TSExtractSeason()
                
            elif self.process.proc.processid.lower()[0:10] == 'resamplets':    
                self.dstNull = self.process.proc.dstcompD[dstcompL[0]]['cellnull']
                self.dstArr = np.ones( ( len(self.dstDateL) ), np.float32)                 
                self.years = len(self.dstDateL)
                self.resampleperiods = self.process.dstperiod.periodstep
    
                if self.process.dstperiod.timestep[-1] in ['D','M'] :
                    #Create an array that stats the start and end columns to resample for each output                     
                    # With daily data just go
                    if self.process.srcperiod.timestep == 'D':
                        self._TSresampleDstep()
                    else:
                        #go via interpolation to daily data
                        #this is a heavy process that takes time
                        self._TSresampleInterpolateDstep()
                elif self.process.dstperiod.timestep == 'A':
                    #Faster then all the others, uses numba
                    self._TSresampleToAnnual()
                else:
                    SNULLE
                    
            elif self.process.proc.processid.lower()[0:7] == 'trendts':
                '''
                movied to intro above
                #The null have to be reset after the original data for trendts
                cellNullEditL = ['avg','std','ts-ic','ols-ic','ols-rmse']
                self.srccomp = srccompL[0]
                self.dstcompL = dstcompL
                self.dstNullD = {}
  
                for item in dstcompL:
                    if item in cellNullEditL:
                        self.process.proc.dstcompD[item]['cellnull'] = self.srcCellNullD[self.srccomp]
                    self.dstNullD[item] = self.process.proc.dstcompD[item]['cellnull']
                print ('self.dstNullD',self.dstNullD)
                self.tsy = np.arange(len(self.process.srcperiod.datumL))
    
                if 'mk' in dstcompL:
                    self.mktsNullA = np.ones(5, np.float32)
                    self.mktsNullA *= self.dstNullD['mk'] 
                    self.mktsItemD = {'mk':0,'ts-mdsl':1,'ts-ic':2,'ts-losl':3,'ts-hisl':4}
                if 'ols' in dstcompL:
                    self.olsArr = np.zeros( ( 4 ), np.float32)
                    self.olsNullA = np.ones(4, np.float32)
                    self.olsNullA *= self.dstNullD['ols'] 
                    self.olsItemD = {'ols':0,'ols-ic':1,'ols-r2':2,'ols-rmse':3}
                '''
                self._TStrendIni()
                
            #elif self.process.proc.processid.lower() in ['seasonfilltsmodissingletile','seasonfilltssmap']:
            elif self.process.proc.processid.lower()[0:12] == 'seasonfillts':
                for comp in srccompL:
                    if comp == 'season':    
                        self.seasonKey = 'season'
                    else:
                        self.mainKey = comp
                self._TSSeasonFill()
            elif self.process.proc.processid.lower()[0:8] == 'autocorr':
                self._AutoCorr()
            elif self.process.proc.processid.lower()[0:15] == 'indexcrosstrend':
                self.srccomp = srccompL[0]
                self.dstcompL = dstcompL
                self.dstNullD = {}
                for item in dstcompL:
                    self.dstNullD[item] = self.process.proc.dstcompD[item]['cellnull']
                     
                self._IndexCrossTrend()
                
            elif self.process.proc.processid.lower()[0:15] == 'imagecrosstrend':
                self.srccomp = srccompL[0]
                self.dstcompL = dstcompL
                self.dstNullD = {}
                for item in dstcompL:
                    self.dstNullD[item] = self.process.proc.dstcompD[item]['cellnull'] 
                #self.dstDatum = self.dstDateD[dstcompL[0]][0]                    
                self._LayerCrossTrend()
                
            elif self.process.proc.processid.lower()[0:15] == 'setassimilation':
                self._SetAssimilation()
                
            elif self.process.proc.processid.lower()[0:10] == 'assimilate':
                for item in dstcompL:
                    self.dstNull = self.process.proc.dstcompD[item]['cellnull']
                if self.allSameSize:
                    self._Assimilate()
                else:
                    self._AssimilateDownscale()
    
            else:
                exitstr = 'Exiting, processid %(p)s missing in ProcessTimeSeries' %{'p':self.process.proc.processid}
                exit(exitstr)
                FISKSOPPA
        #self._CloseDstRasterFiles(locus)
        #self._CloseSrcRasterFiles(locus)
                     
    def _SetReadWriteDimOld(self,locus):
        self.lins = int(self.geoFormatD[locus]['lins'])
        self.cols = int(self.geoFormatD[locus]['cols'])  
        self.imgSize = self.lins*self.cols
        #TGTODO Get the free memory and then set blocksize
        self.blocklines = min(10,self.lins)
        self.blockSize = self.cols*self.blocklines
        self.readitems = int(self.lins/self.blocklines)
        if self.readitems < self.lins/self.blocklines:
            self.lastFullReadItem = self.readitems
            self.lastBlockSize = self.imgSize - self.readitems*self.blockSize
            self.lastBlockLines = int(self.lastBlockSize/self.cols)
            self.readitems += 1
        else:
            self.lastBlockLines = 0
            self.lastFullReadItem  = self.readitems + 10 #never reached
            
    def _SetReadWriteDim(self,locus):
        '''
        '''
        print (self.LayerReaderD)
        #Get min and max lins and cols
        maxlins = 0
        maxcols = 0
        minlins = 999999
        mincols = 999999
        for comp in self.LayerReaderD:
            if self.LayerReaderD[comp]['lins'] > maxlins:
                maxlins = self.LayerReaderD[comp]['lins']
            if self.LayerReaderD[comp]['cols'] > maxcols:
                maxcols = self.LayerReaderD[comp]['cols']
            if self.LayerReaderD[comp]['lins'] < minlins:
                minlins = self.LayerReaderD[comp]['lins']
            if self.LayerReaderD[comp]['cols'] < mincols:
                mincols = self.LayerReaderD[comp]['cols']
            

        if minlins < maxlins:
            
            #the ratio must be an exact integer
            linsratio = float(maxlins)/float(minlins)
            if not linsratio.is_integer():
                exit('The ratio of layers at different resolutions must be an integer')
            colsratio = int(maxcols/mincols) 
            linsratio = int(linsratio)
            if not colsratio == linsratio:
                exit('The ratio of layers at different resolutions must be an integer')  
            blocklines = linsratio
            self.allSameSize = False
        elif mincols < maxcols:
            exit('The ratio of layers at different resolutions must be an integer') 
        else:
            blocklines = min(10,maxlins)
            self.allSameSize = True
            
        self.blocklines = blocklines

        
        #Set the read rules for min, only if min is less than max, otherwise the blockreading collapses
        if not self.allSameSize:
            imgSize, blockSize, readitems, lastBlockLines, lastFullReadItem = self._SetReadBlocks(minlins,mincols,1)
            
            blockratio = int(maxlins/minlins)
            for comp in self.LayerReaderD:
                if self.LayerReaderD[comp]['lins'] == minlins:
                    self.LayerReaderD[comp]['imgsize'] = imgSize
                    self.LayerReaderD[comp]['blocklines'] = 1
                    self.LayerReaderD[comp]['blocksize'] = blockSize
                    self.LayerReaderD[comp]['blockratio'] = blockratio
                    self.LayerReaderD[comp]['readitems'] = readitems
                    self.LayerReaderD[comp]['lastBlockLines'] = lastBlockLines
                    lastBlockSize = imgSize - (readitems-1)*blockSize
                    self.LayerReaderD[comp]['lastBlockSize'] = lastBlockSize
                    #self.LayerReaderD[comp]['lastFullReadItem'] = lastFullReadItem
        #print (self.LayerReaderD)

        #Set the read rules for max
        imgSize, blockSize, readitems, lastBlockLines, lastFullReadItem = self._SetReadBlocks(maxlins,maxcols,blocklines)
        self.readitems = readitems
        self.lastFullReadItem = lastFullReadItem
        blockratio = 1
        for comp in self.LayerReaderD:
            if self.LayerReaderD[comp]['lins'] == maxlins:
                self.LayerReaderD[comp]['imgsize'] = imgSize
                self.LayerReaderD[comp]['blocklines'] = blocklines
                self.LayerReaderD[comp]['blocksize'] = blockSize
                
                self.LayerReaderD[comp]['blockratio'] = blockratio
                self.LayerReaderD[comp]['readitems'] = readitems
                self.LayerReaderD[comp]['lastBlockLines'] = lastBlockLines
                self.LayerReaderD[comp]['lastFullReadItem'] = lastFullReadItem
                lastBlockSize = imgSize - (readitems-1)*blockSize
                print  ('imgSize',imgSize)
                print ('readitems',readitems)
                print ('blockSize',blockSize)
                print ('readitems*blockSize',readitems*blockSize)
                print ('lastBlockSize',lastBlockSize)

                self.LayerReaderD[comp]['lastBlockSize'] = lastBlockSize
                #Set the destimation geoformat after the largest input
                self.geoDstFormatD = self.geoFormatD[comp]
            
    
    def _SetReadBlocks(self,lins,cols,blocklines):
        imgSize =  lins*cols
        blockSize = cols*blocklines
        
        readitems = int(lins/blocklines)
        
        if readitems < lins/blocklines:
            lastFullReadItem = readitems
            lastBlockSize = imgSize - readitems*blockSize
            lastBlockLines = int(lastBlockSize/cols)
            
            readitems += 1
        else:
            lastBlockLines = 0
            lastFullReadItem  = readitems + blocklines #never reached
        return (imgSize, blockSize, readitems, lastBlockLines, lastFullReadItem)
           
    def _CreateDstNullArrays(self,dstcompL,n=0):
        '''Create null arrays for each destination layer
        '''
        self.dstNullDarr = {}
        for dstcomp in dstcompL:
            cellnull = self.process.proc.dstcompD[dstcomp]['cellnull']
            if not n:
                n = len(self.dstDateD[dstcomp])
            self.dstNullDarr[dstcomp] = np.ones(n, np.float32)
            self.dstNullDarr[dstcomp] *= cellnull
                  
    def _ReadSrcBlock(self,locus,comp,blockSize,blockLines):
        '''Blockread the source data
        '''
        #all processing as Float 32
        self.srcRD[comp] = np.ones( ( len(self.process.srcperiod.datumL), blockSize), np.float32)
        self.srcRD[comp] *= self.srcCellNullD[comp]
        x = 0

        for datum in self.process.srcLayerD[locus]:
            if not self.process.srcLayerD[locus][datum]:
                #Fill in with null- no null already set
                print ('    missing datum', datum)
                NOLONGER #missing layer = False set at comp level instead
            else:  
            
                if self.process.srcLayerD[locus][datum][comp]:
                    if self.process.srcLayerD[locus][datum][comp].copy:
                        copydatum = self.AcqDateStrSrcLayerD[datum]
                        srcDatum = self.LayerForAcqDateStrD[copydatum]
                        self.srcRD[comp][x] = self.process.srcLayerD[locus][srcDatum][comp].layer.NPBLOCK
                    else:

                        startlin = int(self.wl/self.LayerReaderD[comp]['blockratio'])
                        readcols = self.LayerReaderD[comp]['cols']
                        #self.process.srcLayerD[locus][datum][comp].layer.ReadBlock(0,self.wl,self.cols,self.blocklines)
                        self.process.srcLayerD[locus][datum][comp].layer.ReadBlock(0,startlin,readcols,blockLines)
                        #print (comp,startlin,readcols,blocklines)
                        #print (self.process.srcLayerD[locus][datum][comp].layer.NPBLOCK.shape)
    
                        self.srcRD[comp][x] = self.process.srcLayerD[locus][datum][comp].layer.NPBLOCK
                        '''
                        print (self.srcRD[comp][x])
                        print (self.srcRD[comp][x].shape)
                        print ('readcols',readcols)
                        print ('blocklines',blocklines)
                        SNULLE
                        '''
                else:
                    pass
                    #print ('no layer file, nodata prevails', locus, datum, comp)
                x += 1

    def _CheckDstRasterFiles(self):
        '''Check if the destination layers already exist
        '''
        self.SkipLocus = True
        for datum in self.process.dstLayerD[self.locus]:
            for comp in self.process.dstLayerD[self.locus][datum]:                
                if not self.process.dstLayerD[self.locus][datum][comp]._Exists() or self.process.overwrite:
                    self.SkipLocus = False 
                    return 
                elif path.isfile(self.process.dstLayerD[self.locus][datum][comp].FPN):
                    self.session._InsertLayer(self.process.dstLayerD[self.locus][datum][comp],self.process.overwrite,self.process.delete)
         
    def _CreateOpenDstRasterFiles(self, locus):
        '''Create and open the destimation layers
        '''
        for datum in self.process.dstLayerD[locus]:
            for comp in self.process.dstLayerD[locus][datum]:
                #print ('        creating dstcomp',comp,datum,self.process.dstLayerD[locus][datum][comp].FPN)
                #Transfer the geoformat for this locus
                for item in self.geoDstFormatD:
                    setattr(self.process.dstLayerD[locus][datum][comp], item, self.geoDstFormatD[item])  
                if not self.process.dstLayerD[locus][datum][comp]._Exists() or self.process.overwrite:
                    #TGTODO THIS IS A FUNCTION OF THE LAYER!!
                    self.process.dstLayerD[locus][datum][comp].dstDS = mj_gis.RasterCreateWithFirstLayer(self.process.dstLayerD[locus][datum][comp].FPN, self.process.dstLayerD[locus][datum][comp])
                    self.dstDateD[comp].append(datum)   
                else:
                    #print ('insert',self.process.dstLayerD[locus][datum][comp].FPN)
                    self.session._InsertLayer(self.process.dstLayerD[locus][datum][comp],self.process.overwrite,self.process.delete)
                    self.process.dstLayerD[locus][datum][comp] = False
                    self.dstDateD[comp].append(False)
                    
    def _CloseDstRasterFiles(self, locus):
        '''Close the created destination layers
        '''
        for datum in self.process.dstLayerD[locus]:
            for comp in self.process.dstLayerD[locus][datum]:
                if self.process.dstLayerD[locus][datum][comp]:
                    print ('closing',self.process.dstLayerD[locus][datum][comp].FPN)
                    self.process.dstLayerD[locus][datum][comp].dstDS.CloseDS()
                    self.session._InsertLayer(self.process.dstLayerD[locus][datum][comp],self.process.overwrite,self.process.delete)
                    self.process.dstLayerD[locus][datum][comp] = None
           
    def _SetNumbaArrayReaderOld(self,locus):
        '''Creates a dict that list which np array columns should be resampled to which output
        '''
        self.arrIO = {}
        self.numbaStartCol = []
        self.numbaEndCol = []
        self.numbaCols = []
        
        if self.process.srcperiod.timestep == 'D':
            #self.srcAcqDateL = [mj_dt.yyyymmddDate(item) for item in self.process.srcperiod.datumL]
            srcStrDateL= self.srcDateL
            srcDateL = [mj_dt.yyyymmddDate(item) for item in self.process.srcperiod.datumL]
        elif self.process.srcperiod.timestep[-1] == 'D':
            srcAcqDateL = [mj_dt.yyyydoyDate(item) for item in self.process.srcperiod.datumL]
            srcStrDateL = [mj_dt.DateToStrDate(item) for item in srcAcqDateL]
            firstYYYYDOY = self.process.srcperiod.datumL[0]
            lastYYYYDOY = self.process.srcperiod.datumL[-1]
            firstdate = mj_dt.yyyydoyDate(firstYYYYDOY)
            lastdate = mj_dt.yyyydoyDate(lastYYYYDOY)

            srcDateL = mj_dt.DateRange(firstdate,lastdate)

        else:      
            BALLE
            
        pddates = to_datetime(srcStrDateL, format='%Y%m%d')
        self.pddates = pddates.to_pydatetime()
           
        AcqSteps = []
        for d in pddates:
            AcqSteps.append(mj_dt.DateDiff(d,pddates[0])) 
        self.AcqSteps = np.array( AcqSteps )
        self.interpolSteps = np.array([j-i for i, j in zip(self.AcqSteps[:-1], self.AcqSteps[1:])])
        
        self.interpolA = np.zeros( ( len(srcDateL) ), np.float32)

        if self.process.dstperiod.timestep == 'M':
            for y,dstDate in enumerate(self.dstDateL):
                if self.process.dstperiod.datumD[dstDate]['acqdate']:
                    #For each date there is one starting and ending from the src data
                    startdate = self.process.dstperiod.datumD[dstDate]['acqdate']
                    
                    enddate = mj_dt.AddMonth(startdate, 1)
                    #remove one day
                    enddate = mj_dt.DeltaTime(enddate, -1)
                    
                    startflag = False
                    endflag = False
                    #XCheck which columns in the input array will belong to this output date!
                    for col,datum in enumerate(srcDateL):
                        end = col
                        if datum >= startdate and not startflag:
                            start = col  
                            startflag = True
                        if datum >= enddate:
                            endflag = True
                            break
                    if not endflag:
                        '''
                        CHECKERROR
                        for date in self.srcDateDateL:
                            print (date)

                        print ('end',end)
                        '''
                        if self.process.proc.acceptmissing:
                            endflag = True
                    if not startflag:
                        CHECKERROR
                        
                    if startflag and endflag:
                        self.arrIO[y] = {'start':start,'end':end, 'items':end-start+1}
                        self.numbaStartCol.append(start)
                        self.numbaEndCol.append(end+1)
                        cols = [y,start,end+1]
                        self.numbaCols.append(cols)
        elif self.process.dstperiod.timestep[-1]  == 'D':
            deltadays = self.process.dstperiod.periodstep
        
            for y,dstDate in enumerate(self.dstDateL):
                if self.process.dstperiod.datumD[dstDate]['acqdate']:
                    #For each date there is one starting and ending from the src data
                    startdate = self.process.dstperiod.datumD[dstDate]['acqdate']
                    
                    enddate = mj_dt.DeltaTime(startdate, deltadays-1)
                    startflag = False
                    endflag = False
                    #XCheck which columns in the input array will belong to this output date!
                    for col,datum in enumerate(srcDateL):
                        end = col
                        if datum >= startdate and not startflag:
                            start = col  
                            startflag = True
                        if datum >= enddate:
                            endflag = True
                            break
                    if not endflag:
                        print (datum, 'start',start,'end',end,'items',end-start+1)
                        print (srcDateL[start:end])
 
                        CHECKERROR
                    if not startflag:
                        CHECKERROR
                    if startflag and endflag:
                        self.arrIO[y] = {'start':start,'end':end, 'items':end-start+1}
                        self.numbaStartCol.append(start)
                        self.numbaEndCol.append(end+1)
                        cols = [y,start,end+1]
                        self.numbaCols.append(cols)
                    print ('setting numbaCols',datum,cols,end-start+1,srcDateL[start:end])
        else:
            exit('Unknown timestep in SetNumbaArrayReader (timeseries.timeseries.py',self.process.dstperiod.timestep)
        print (cols)
        for cols in self.numbaCols:
            print (cols)
          
    def _SetNumbaArrayReader(self,locus):

        '''Creates a dict that list which np array columns should be resampled to which output
        '''
        self.arrIO = {}
        self.numbaStartCol = []
        self.numbaEndCol = []
        self.numbaCols = []
        
        if self.process.srcperiod.timestep == 'D':
            #self.srcAcqDateL = [mj_dt.yyyymmddDate(item) for item in self.process.srcperiod.datumL]
            srcStrDateL= self.srcDateL
            srcDateL = [mj_dt.yyyymmddDate(item) for item in self.process.srcperiod.datumL]
        elif self.process.srcperiod.timestep[-1] == 'D':
            srcAcqDateL = [mj_dt.yyyydoyDate(item) for item in self.process.srcperiod.datumL]
            srcStrDateL = [mj_dt.DateToStrDate(item) for item in srcAcqDateL]
            firstYYYYDOY = self.process.srcperiod.datumL[0]
            lastYYYYDOY = self.process.srcperiod.datumL[-1]
            firstdate = mj_dt.yyyydoyDate(firstYYYYDOY)
            lastdate = mj_dt.yyyydoyDate(lastYYYYDOY)

            srcDateL = mj_dt.DateRange(firstdate,lastdate)

        else:      
            SNULLEBULLE
            
        pddates = to_datetime(srcStrDateL, format='%Y%m%d')
        self.pddates = pddates.to_pydatetime()
           
        AcqSteps = []
        for d in pddates:
            AcqSteps.append(mj_dt.DateDiff(d,pddates[0])) 
        self.AcqSteps = np.array( AcqSteps )
        self.interpolSteps = np.array([j-i for i, j in zip(self.AcqSteps[:-1], self.AcqSteps[1:])])
        
        self.interpolA = np.zeros( ( len(srcDateL) ), np.float32)

        if self.process.dstperiod.timestep == 'M':
            for y,dstDate in enumerate(self.dstDateL):
                if self.process.dstperiod.datumD[dstDate]['acqdate']:
                    #For each date there is one starting and ending from the src data
                    startdate = self.process.dstperiod.datumD[dstDate]['acqdate']
                    
                    enddate = mj_dt.AddMonth(startdate, 1)
                    #remove one day
                    enddate = mj_dt.DeltaTime(enddate, -1)
                    
                    startflag = False
                    endflag = False
                    #XCheck which columns in the input array will belong to this output date!
                    for col,datum in enumerate(srcDateL):
                        end = col
                        if datum >= startdate and not startflag:
                            start = col  
                            startflag = True
                        if datum >= enddate:
                            endflag = True
                            break
                    if not endflag:
                        '''
                        CHECKERROR
                        for date in self.srcDateDateL:
                            print (date)

                        print ('end',end)
                        '''
                        if self.process.proc.acceptmissing:
                            endflag = True
                    if not startflag:
                        CHECKERROR
                        
                    if startflag and endflag:
                        self.arrIO[y] = {'start':start,'end':end, 'items':end-start+1}
                        self.numbaStartCol.append(start)
                        self.numbaEndCol.append(end+1)
                        cols = [y,start,end+1]
                        self.numbaCols.append(cols)
        elif self.process.dstperiod.timestep[-1]  == 'D': 
  
            for i,d in enumerate(self.process.dstperiod.datumD):
                firstdate = self.process.dstperiod.datumD[d]['firstdate']
                lastdate = self.process.dstperiod.datumD[d]['lastdate']
                acqdatestr = self.process.dstperiod.datumD[d]['acqdatestr']
                #Loop over the srcdata to get start and end as columns
                firstcol = srcDateL.index(firstdate)
                if lastdate in srcDateL:
                    lastcol = srcDateL.index(lastdate)
                    self.arrIO[i] = {'start':firstcol,'end':lastcol, 'items':lastcol-firstcol+1}
                    self.numbaStartCol.append(firstcol)
                    self.numbaEndCol.append(lastcol+1)
                    cols = [i,firstcol,lastcol+1]
                    self.numbaCols.append(cols)
                    print ('setting numbaCols',acqdatestr,cols,lastcol-firstcol+1,srcDateL[firstcol:lastcol+1])

                else:
                    print ('date can not be produced',acqdatestr,d)
                    #pop datum from date list and from locus
                    self.process.dstperiod.datumD.pop(d, None)
                    self.process.dstLayerD[locus].pop(d,None)
                    return

        else:
            exit('Unknown timestep in SetNumbaArrayReader (timeseries.timeseries.py',self.process.dstperiod.timestep)
        print (cols)
        for cols in self.numbaCols:
            print (cols)   
              
    def _TSFillNoData(self):
        '''Fill timeseries nodata by interpolation
        '''
        R = self.srcRD[self.srccomp]       
        Rr = np.apply_along_axis( self._FillAlongAxis, 0, R )
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                
    def _AverageTSPerDate(self):
        '''Calculate timeseries average per season
        '''
        #nanmean not supported by numba at time of writing
        #from geoimagine.ktnumba import AverageTSPerDateNumba
        for x,comp in enumerate(self.srccompL):
            if x == 0:
                R = np.copy(self.srcRD[comp])
            else:
                R = np.dstack((R,self.srcRD[comp]))
        Rr = np.nanmean( R, axis=2 )
        Rr[np.isnan(Rr)] = self.dstNull
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
        
    def _TSExtractSeason(self):
        '''Extract timeseries seasonal signal
        '''
        R = self.srcRD[self.srccomp]  
        Rr = np.apply_along_axis( self._ExtractSeason, 0, R )
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
           
    def _TSSeasonFill(self):  
        '''Interpolate null in seasonal data
        '''
        ts = self.srcRD[self.mainKey] 
        season = self.srcRD[self.seasonKey]

        result = np.empty_like(ts.T)
        for i,(x,y) in enumerate(zip(ts.T,season.T)):
            result[i] = self._SeasonFill(x,y)
        for x,row in enumerate(result.T):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                         
    def _TSresampleDstep(self):
        '''Resample timeseries with daily frequency
        '''
        R = self.srcRD[self.srccomp]  
                
        if self.process.params.method == 'avg':           
            Rr = np.apply_along_axis( self._ResampleDictPeriodAvg, 0, R )
        elif self.process.params.method == 'sum':
            Rr = np.apply_along_axis( self._ResampleDictPeriodSum, 0, R )
        else:
            exit('unrecoginized resample functon in _TSresampleDstep (timeseries.timeseries.py)',self.process.params.method)  
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                
    def _TSresampleInterpolateDstep(self):
        '''Resample timeseries temporal frequency
        '''
        R = self.srcRD[self.srccomp]  
          
        if self.process.params.method == 'avg':           
            Rr = np.apply_along_axis( self._ResampleInterpolPeriodAvg, 0, R )
        elif self.process.params.method == 'sum':
            Rr = np.apply_along_axis( self._ResampleInterpolPeriodSum, 0, R )
        else:
            exit('unrecoginized resample functon in _TSresampleDstep (timeseries.timeseries.py)',self.process.params.method)  
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                    
    def _TSresampleToAnnual(self):
        '''Resample timeseries temporal frequency
        '''
        R = self.srcRD[self.srccomp]  
        if self.process.params.method == 'avg':           
            Rr = np.apply_along_axis( self._ResampleToAnnualAvg, 0, R )
        elif self.process.params.method == 'sum':
            Rr = np.apply_along_axis( self._ResampleToAnnualSum, 0, R )
        else:
            exit('unrecoginized resample functon in _TSresample (timeseries.timeseries.py)',self.process.params.method)  
        for x,row in enumerate(Rr):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                
    def _TStrendIni(self):
        '''Initiate calculation of time series statistics
        '''
        ts = self.srcRD[self.srccomp]
        self._TStrend(ts)
          
    def _TStrend(self,ts):
        '''Calculation of time series statistics
        '''
        if 'avg' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['avg']:
            avg = np.apply_along_axis( self._OverAllAvg, 0, ts )
            self._WriteBlock(avg,self.dstDatum,'avg')

        if 'std' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['std']:
            std = np.apply_along_axis( self._OverAllStd, 0, ts )
            self._WriteBlock(std,self.dstDatum,'std')
            
        if 'cov' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['cov']:
            cov = np.apply_along_axis( self._OverAllCov, 0, ts )
            cov[cov < -32768] = -32768
            cov[cov > 32768] = 32768
            self._WriteBlock(cov,self.dstDatum,'cov')
    
        if 'acv' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['acv']:
            anncov = np.apply_along_axis( self._InterAnnualCov, 0, ts )
            anncov[anncov < -32768] = -32768
            anncov[anncov > 32768] = 32768
            self._WriteBlock(anncov,self.dstDatum,'anncov')
            
        if 'mcv' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['mcv']:
            monthcov = np.apply_along_axis( self._SeasonalCov, 0, ts )
            monthcov[monthcov < -32768] = -32768
            monthcov[monthcov > 32768] = 32768
            self._WriteBlock(monthcov,self.dstDatum,'monthcov')
 
        if 'mk' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['mk']:
            self.mktsA = np.zeros( [5 , ts.shape[1] ], np.float32)
            self.counter = -1
            dummy = np.apply_along_axis( self._MKtestAlongAxis, 0, ts ) 
            for item in self.mktsItemD:
                self._WriteBlock(self.mktsA[self.mktsItemD[item]],self.dstDatum,item)
                
        if 'ols' in self.dstcompL and self.process.dstLayerD[self.locus][self.dstDatum]['ols']:
            self.olsA = np.zeros( [4 , ts.shape[1] ], np.float32)
            self.counter = -1
            dummy = np.apply_along_axis( self._OLSAlongAxis, 0, ts ) 

            for item in self.olsItemD:
                self._WriteBlock(self.olsA[self.olsItemD[item]],self.dstDatum,item)
           
    def _WriteBlock(self,row,datum,dstcomp):
        if self.blocklines == 1:
            R2D = np.atleast_2d(row)
        else:
            R2D = np.reshape(row, (self.blocklines,-1))

        self.process.dstLayerD[self.locus][datum][dstcomp].dstDS.WriteBlock(0,self.wl,R2D)
                
    def _AutoCorr(self):
        R = self.srcRD[self.srccomp]   
        self.lagL = []
        for b in range( self.process.params.nlags):
            self.lagL.append([])
        if self.process.params.resampleseasonal:
            #lag0 is just a dummy, the data is saveid in self.lagL
            lag0 = np.apply_along_axis( self._SeasonalAutoCorr, 0, R, partial=self.process.params.partial) 
        else:
            #lag0 is just a dummy, the data is saveid in self.lagL
            lag0 = np.apply_along_axis( self._FullAutoCorr, 0, R, partial=self.process.params.partial)
        #skip the first lag
        for b in range(1,self.process.params.nlags):  
            row = np.array( self.lagL[b] )
            row.round(decimals=6)  
            if self.blocklines == 1:
                R2D = np.atleast_2d(row)
            else:
                R2D = np.reshape(row, (self.blocklines,-1))              
            datum = self.dstDateL[b-1]
            self.process.dstLayerD[self.locus][datum][self.dstcomp].dstDS.WriteBlock(0,self.wl,R2D)
                
    def _IndexCrossTrend(self):
        self.allcrosscompsL = ['pearson','lag']
        R = self.srcRD[self.srccomp]
        self.dodecompose = False
        #Create empty lists for holding decomposed data
        if self.process.params.xcrossobserved:
            self.obs = []

        if self.process.params.xcrosstendency:
            self.dodecompose = True
            self.tendency = []
        if self.process.params.xcrosseason:
            self.dodecompose = True
            self.fullseasons = [] 
        if self.process.params.xcrossresidual:
            self.dodecompose = True
            self.residual = []
        '''
        #Create empty lists for holding decomposed data
        self.tendency = []
        self.obs = []
        self.fullseasons = [] 
        self.residual = []
        self.seasons = []
        '''
        #Set the dict for storing the results
        self.xcross ={}
        for i in self.indexD:
            for x in self.process.proc.paramsD['xcrosscompsL']:
                for y in self.allcrosscompsL:
                    band ='%(x)s-%(l)s-%(i)s' %{'x':x, 'l':y,'i':i}
                    self.xcross[band] = []                   
        dummy = np.apply_along_axis( self._IndexCrossTrendsAlongAxis, 0, R )
       
        self._IndexCompTrends()
        
    def _LayerCrossTrend(self):
        self.allcrosscompsL = ['pearson','lag']
        MASTER = self.srcRD[self.idD['master']]
        SLAVE = self.srcRD[self.idD['slave']]
        #The setting of dodecompose should be moved to above
        self.dodecompose = False
        #Create empty lists for holding decomposed data
        if self.process.params.xcrossobserved:
            self.obs_master = []
            self.obs_slave = []
        if self.process.params.xcrosstendency:
            self.dodecompose = True
            self.tendency_master = []
            self.tendency_slave = []
        if self.process.params.xcrosseason:
            self.dodecompose = True
            self.fullseasons_master = [] 
            self.fullseasons_slave = []
        if self.process.params.xcrossresidual:
            self.dodecompose = True
            self.residual_master = []
            self.residual_slave = []
        
        #self.seasons_master = []
        #self.seasons_slave = []
        
        #Set the dict for storing the results
        self.xcross ={}
        
        for x in self.process.proc.paramsD['xcrosscompsL']:
            for y in self.allcrosscompsL:
                band ='%(x)s-%(l)s' %{'x':x, 'l':y}
                self.xcross[band] = []  

        for col in range(MASTER.shape[1]):
            m = MASTER[:,col]
            if np.isnan(np.sum(m)):

                self._LayerXCorrSetTStoNull()
                continue
            s = SLAVE[:,col]
            if np.isnan(np.sum(s)):

                self._LayerXCorrSetTStoNull()
                continue

            self._LayerXCorrDecomposeTS(m,s)
        self._LayerCompTrends()
          
    def _IndexCompTrends(self):
        '''
        '''
        for x in self.process.proc.paramsD['xcrosscompsL']: 
            if x == 'season':
                A = np.transpose(np.array(self.fullseasons))  
            elif x == 'residual':
                A = np.transpose(np.array(self.residual))
            elif x == 'obs':
                A = np.transpose(np.array(self.obs))
            elif x == 'tendency':  
                A = np.transpose(np.array(self.tendency))
            else:
                print (x)
                exit('no such decomposition component',x)

            for i in self.indexD:
                self.xcross = {}
                self.xcross['lag'] = []
                self.xcross['pearson'] = []
                if self.process.params.normalize:
                    self.Inorm = ( self.indexD[i][x] - np.average(self.indexD[i][x]) ) /  (np.std(self.indexD[i][x]))
                else:
                    self.Inorm = self.indexD[i][x]
                #apply_along_axis returns a dummy, the values are stored in the self.xcross dict
                dummy = np.apply_along_axis( self._IndexXCorr, 0, A )
                        
                for y in self.process.proc.paramsD['xcrossdstL']:

                    band ='%(x)s-%(l)s-%(i)s' %{'x':x, 'l':y,'i':i} 
 
                    if self.process.dstLayerD[self.locus][self.dstDatum][band]:
                        row = np.array(self.xcross[y])                        
                        self._WriteBlock(row,self.dstDatum,band)
                   
    def _LayerCompTrends(self):
        '''
        '''
        for x in self.process.proc.paramsD['xcrosscompsL']:
            if x == 'obs':
                M = np.transpose(np.vstack(self.obs_master))
                S = np.transpose(np.vstack(self.obs_slave)) 
            elif x == 'season':
                M = np.transpose(np.vstack(self.fullseasons_master))  
                S = np.transpose(np.vstack(self.fullseasons_slave))
            elif x == 'residual':
                M = np.transpose(np.vstack(self.residual_master))
                S = np.transpose(np.vstack(self.residual_slave))
            elif x == 'tendency':  
                M = np.transpose(np.vstack(self.tendency_master))
                S = np.transpose(np.vstack(self.tendency_slave))
            else:
                exit('no such decomposition component',x)

            self.xcross = {}
            self.xcross['lag'] = []
            self.xcross['pearson'] = []
 
            for col in range(M.shape[1]):
                mArr = M[:,col]                    
                sArr = S[:,col]
                '''
                print ('column',col)
                print ('mArr',mArr[0:10])
                print ('sArr',sArr[0:10])
                '''
                self._LayerXCorr(mArr,sArr)
            
            for y in self.process.proc.paramsD['xcrossdstL']:
                band ='%(x)s-%(l)s' %{'x':x, 'l':y} 
                row = np.array(self.xcross[y])

                if self.blocklines == 1:
                    R2D = np.atleast_2d(row)
                else:
                    R2D = np.reshape(row, (self.blocklines,-1))
                self.process.dstLayerD[self.locus][self.dstDatum][band].dstDS.WriteBlock(0,self.wl,R2D)
            
            #And then  run the fixed analysis
            for lag in self.process.proc.paramsD['xcrossLagL']:
                self.xcorr = []
                for col in range(M.shape[1]):
                    mArr = M[:,col]                    
                    sArr = S[:,col]
                    self._LayerCrossCorrFixedLag(mArr,sArr,lag)
                
                band ='%(x)s-pearson-lag%(l)s' %{'x':x, 'l':lag} 
                row = np.array(self.xcorr)
                if self.blocklines == 1:
                    R2D = np.atleast_2d(row)
                else:
                    R2D = np.reshape(row, (self.blocklines,-1))
                self.process.dstLayerD[self.locus][self.dstDatum][band].dstDS.WriteBlock(0,self.wl,R2D)
   
    def _SetAssimilation(self):
        MASTER = self.srcRD[self.idD['master']]
        if self.process.params.scalefac != 1:
            MASTER *= self.process.params.scalefac
        if self.process.params.offsetadd:
            MASTER += self.process.params.offsetadd
        SLAVE = self.srcRD[self.idD['slave']]

        mstAvg = np.apply_along_axis( self._OverAllNanMean, 0, MASTER )
        mstStd = np.apply_along_axis( self._OverAllNanStd, 0, MASTER )
        slvAvg = np.apply_along_axis( self._OverAllNanMean, 0, SLAVE )
        slvStd = np.apply_along_axis( self._OverAllNanStd, 0, SLAVE )
        stdrat = mstStd/slvStd

        stdrat[np.isnan(stdrat)] = self.dstNullD['stdrat']
        stdrat[np.isnan(mstAvg)] = self.dstNullD['stdrat']
        stdrat[np.isnan(slvAvg)] = self.dstNullD['stdrat']
        stdrat[mstStd <= 0] = self.dstNullD['stdrat']
        stdrat[slvStd <= 0] = self.dstNullD['stdrat']

        #write to files
        mstAvg[np.isnan(mstAvg)] = self.dstNullD['mstavg']
        self._WriteBlock(mstAvg,self.dstDateL[0],'mstavg' )
                
        slvAvg[np.isnan(slvAvg)] = self.dstNullD['slvavg']
        self._WriteBlock(slvAvg,self.dstDateL[0],'slvavg' )
             
        self._WriteBlock(stdrat,self.dstDateL[0],'stdrat' )

    
    def _Assimilate(self):
        SLAVE = self.srcRD[self.idD['slave']]
        #self.mstAvg = self.srcRD[self.idD['mstavg']][:,0]  
        #self.slvAvg = self.srcRD[self.idD['slvavg']][:,0]
        #self.stdrat = self.srcRD[self.idD['stdrat']][:,0]

        self.mstAvg = self.srcRD[self.idD['mstavg']][0,:]  
        self.slvAvg = self.srcRD[self.idD['slvavg']][0,:]
        self.stdrat = self.srcRD[self.idD['stdrat']][0,:]
        self.counter = 0

        ASSIM = np.apply_along_axis( self._AssimAlongAxis, 0, SLAVE )
        #Set the dstnull
        ASSIM[np.isnan(SLAVE)] = self.dstNull
        for x,row in enumerate(ASSIM):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)
                    
    def _AssimAlongAxis(self, ts):
        '''
        '''
        
        if np.isnan(self.stdrat[self.counter]):
            self.counter += 1
            return ts
        
        if self.kernel:
            ts = np.convolve(ts, self.kernel, mode='same')
        
        a = ts-self.slvAvg[self.counter]
        #multiply with std ratio factor
        a = a*self.stdrat[self.counter]
        #add the master mean
        a = a + self.mstAvg[self.counter]
        #add one to counter to get to next cell
        if self.process.params.assimfrac < 1.0:
            a = a*self.process.params.assimfrac+ts*(1-self.process.params.assimfrac)
        if self.threshold:
            a = np.where(ts > self.maxthreshold,ts,a)
            a = np.where(ts < self.minthreshold,ts,a)
            #a[ts > self.maxthreshold] = ts
            #a[ts < self.minthreshold] = ts
        a[a < self.process.params.dstmin] = self.process.params.dstmin
        a[a > self.process.params.dstmax] = self.process.params.dstmax
        self.counter += 1

        return a

    def _AssimilateDownscale(self):
        SLAVE = self.srcRD[self.idD['slave']]
        self.mstAvg = self.srcRD[self.idD['mstavg']][0,:]  
        self.slvAvg = self.srcRD[self.idD['slvavg']][0,:]
        self.stdrat = self.srcRD[self.idD['stdrat']][0,:]

        ASSIM = np.apply_along_axis( self._AssimDownscaleAlongAxis, 1, SLAVE )

        #Set the dstnull
        ASSIM[np.isnan(SLAVE)] = self.dstNull
        for x,row in enumerate(ASSIM):
            datum = self.dstDateL[x]
            if datum:
                self._WriteBlock(row,datum,self.dstcomp)

    def _AssimDownscaleAlongAxis(self,region):
        '''This is done per region, not per date
        '''
        cell = region.reshape(-1,self.LayerReaderD[self.idD['slave']]['cols'])
        assim = cell.copy()

        l = self.LayerReaderD[self.idD['slave']]['blocklines']

        for col in range(self.LayerReaderD[self.idD['slave']]['readitems']):

            c = col*l
            #rect = np.copy(cell[c:c+l,0:l])
            rect = np.copy(cell[0:l,c:c+l])
            if np.isnan(self.stdrat[col]):
                #assim[c:c+l,0:l] = rect
                assim[0:l,c:c+l] = rect
            else:
                #remove the slave mean
                a = rect-self.slvAvg[col]
                #multiply with std ratio factor
                a = a*self.stdrat[col]
                #add the master mean
                a = a + self.mstAvg[col]
                #Fill in the downscaled data
                assim[0:l,c:c+l] = a

        if self.process.params.assimfrac < 1.0:
            assim = assim*self.process.params.assimfrac+cell*(1-self.process.params.assimfrac)
        if self.threshold:
            assim = np.where(cell > self.maxthreshold,cell,assim)
            assim = np.where(cell < self.minthreshold,cell,assim)
        assim[assim < self.process.params.dstmin] = self.process.params.dstmin
        assim[assim > self.process.params.dstmax] = self.process.params.dstmax
        
        return assim.flatten()
