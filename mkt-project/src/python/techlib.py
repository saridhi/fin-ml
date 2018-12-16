# -*- coding: cp1252 -*-
import sys
import datetime
from numpy import std
import array
from timeseries import TimeSeries
import stats

USECOM = False

# Helper class that includes Technical Analysis functions and returns values associated with the indicators
# When comparing TimeSerie objects the code assumes that the O, H, L, C vectors are aligned in dates and values
# Author: Dhiren Sarin

def ConvertDate( value ):
    if type(value) == datetime.date:
        return value
    elif type(value) == datetime.datetime:
        return value.date()
    elif type(value) == pywintypes.TimeType:
        def ole2datetime(oledt):
            return datetime.datetime(1899, 12, 30, 0, 0, 0) + datetime.timedelta(days=float(oledt))
        return ole2datetime(value).date()
    else:
        raise ValueError('Cannot be casted into a date!')
    
def QAAddTenorToDate(refdate=datetime.date.today(), tenor='3m'):

    import qadefs

    qaRefDate = QACOM.Date().createFromYearMonthDay(refdate.year, refdate.month, refdate.day, None)

    qaHolidays = QACOM.HolidayList().createFromSerials('weekends', [], qadefs.MaskWeekend)

    qaExpiryDate = QACOM.TenorUtils().addTenorToDate(qaRefDate, tenor, qadefs.RollForward, qaHolidays)
    
    return datetime.date(qaExpiryDate.getYear, qaExpiryDate.getMonth, qaExpiryDate.getDay)


class TechLib():

        _reg_clsid_ = "{64596F55-89D3-42D3-B621-6EE13C2B1956}" # Created from pythoncom.CreateGuid()
        _reg_desc_ = "Python TechLib"
        _reg_progid_ = "Python.TechLib"
        
        """
        #Constructor
        def __init__(self):
                return None
        """
        
        def truncateADX(self,rollingADX,nbBizDaysAgo = 0):
            
            if nbBizDaysAgo == 0:
                adx = rollingADX
                
                
            else:
                adx = Ether.Creator('TimeSerie', Dates=rollingADX.Dates()[:-nbBizDaysAgo], values=rollingADX.Values()[:-nbBizDaysAgo])
                
            return adx

        # truncate vector of slope values and dates depending on value of nbBizDaysAgo
        
        def truncateSlope(self,securityC):
            
            slope = self.rRegSlope(securityC, 5)
            return slope

        
        def truncateTechSeries(self,nbBizDaysAgo = 0):
            if nbBizDaysAgo == 0:
                securityO = self.securityO()
                securityH = self.securityH()
                securityL = self.securityL()
                securityC = self.securityC()
                
            else:
                securityO = TimeSeries(dates=self.securityO().Dates()[:-nbBizDaysAgo], values=self.securityO().Values()[:-nbBizDaysAgo])
                securityH = TimeSeries(dates=self.securityH().Dates()[:-nbBizDaysAgo], values=self.securityH().Values()[:-nbBizDaysAgo])
                securityL = TimeSeries(dates=self.securityL().Dates()[:-nbBizDaysAgo], values=self.securityL().Values()[:-nbBizDaysAgo])
                securityC = TimeSeries(dates=self.securityC().Dates()[:-nbBizDaysAgo], values=self.securityC().Values()[:-nbBizDaysAgo])
                                
            return securityO,securityH,securityL,securityC

        # Slope using least sq method
        def rRegSlope(self, tscom, lag = 5):

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = list(ts.Values())

                lowind = 0
                for i in (range(0,len(Dates))):
                    datelag = Dates[i] - datetime.timedelta(lag)
                    if datelag > Dates[0]:                
                        j = 0
                        for ind, d in zip(range(lowind,i),Dates[lowind:i]):
                            if d<datelag:
                                j = ind
                        lowind = j

                        y=Values[(i-lag+1):i+1]
                        if y!=[]:
                                mdates.append( Dates[i] )
                                x=range(1,(lag+1))
                                length = len(Values)
                                gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                                mvalues.append( gradient )

                return TimeSeries(dates = mdates, values = mvalues)
                

        # Relative Strength Index
        
        def rRsi(self, n = 9): 
                mvalues = []
                mdates =[]
                pnl = self.securityC().Changes(False)
                gains = pnl.Floor(0.)
                losses = (pnl*(-1.)).Floor(0.)
                gainsExpMean = gains.RExpMean(n)
                lossesExpMean = losses.RExpMean(n)
                for g,l,d in zip(gainsExpMean.Values(), lossesExpMean.Values(), gainsExpMean.Dates()):
                        if (l!=0): #check for division by zero because TimeSeries object does not do this!
                                mvalues.append(g/l)
                                mdates.append(d)
                hundred = TimeSeries(dates = mdates, values = [100.]*len(mdates))
                rs = TimeSeries(dates = mdates, values = mvalues)
                return hundred - hundred/(rs+1.)

        # True Range to be added into techlib, though to substitue Ether for non-Ether
        # True Range is an indication of price movement/volatility -- http://en.wikipedia.org/wiki/Average_True_Range
        def trueRange(self):
                tr = []
                LoValues = self.securityL().Values()
                HiValues = self.securityH().Values()
                ClValues = self.securityC().Values()
                for i in range(1,len(self.securityC().Dates())):
                    hiValue = HiValues[i]
                    loValue = LoValues[i]
                    truerange = max(abs(hiValue- loValue),abs(hiValue- ClValues[i-1]),abs(ClValues[i-1]- loValue))
                    tr.append(truerange)
                return TimeSeries(dates=self.securityC().Dates(), values=tr)


        # %D on Slow Stochastics
        def rStoD(self, n = 10): 
                return self.rMean(self.rStoK(n), 3)

        # %K on slow Stochastics - Simplified algorithm as per CQG
        
        def rStoK(self, n = 10):

                D = (self.rMax(self.securityH(), n)-self.rMin(self.securityL(), n))
                N = (self.securityC()-self.rMin(self.securityL(), n))
                valK = []
                valD = list(D.Values())
                valN = list(N.Values())
                for i in range(0,len(valD)):
                    if valD[i] ==0:
                        k = 0
                    else:
                        k = ((float(valN[i]))/(valD[i]))
                    valK.append(k)

                K = TimeSeries(dates = N.Dates(), values = valK)
                #K = ((self.securityC()-self.rMin(self.securityL(), n))/(self.rMax(self.securityH(), n)-self.rMin(self.securityL(), n)))*100.
                return self.rMean(K, 3)

        # Rolling mean (Simple moving average)
        def rMean(self, tscom, lag = 21):  
                """ Returns the rolling mean values as a new time-serie """

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                
                def auxmean(ts, start, end):
                    length = end-start+1
                    if (length > 0):
                        return Dates[end],((sum(Values[start:end+1]))/length)
                
                for i in (range(0,len(Dates))):
                        if ((i+lag)>len(Dates)):
                                break
                        dt, val = auxmean(ts, i, (i+lag-1))
                        mdates.append( dt )
                        mvalues.append( val )
                        
                return TimeSeries(dates= mdates, values = mvalues)

        # Simple mean -- not rolling
        def mean(self, tscom, start, end):

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                length = end-start+1
                if (length > 0):
                        return ConvertDate(ts.Dates()[end]),((sum(ts.Values()[start:end+1]))/length)
        
        # Rolling minimum of a time series
        def rMin(self, tscom, lag = 10, roll = True): 
                """ Returns the rolling minimum values as a new time-serie """

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()
                
                length = len(Values)
                if (roll == False): return min(Values[length-lag:length-1])

                for i in range(0,len(Dates)):
                        if (i+lag)>len(Dates):
                                break
                        mdates.append( Dates[i+lag-1] )
                        mvalues.append( min(Values[i:(i+lag)]) )

                return TimeSeries(dates = mdates, values = mvalues)
                

        # Rolling maximum of a time series
        def rMax(self, tscom, lag = 10, roll = True): 
                """ Returns the rolling maximum values as a new time-serie """

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                if (roll == False): return max(Values[length-lag:length-1])
                
                for i in range(0,len(Dates)):
                        if (i+lag)>len(Dates):
                                break
                        mdates.append( Dates[i+lag-1] )
                        mvalues.append( max(Values[i:(i+lag)]) )

                return TimeSeries(dates = mdates, values = mvalues)
                

        # Rolling Rate of Change oscillator
        
        def rROC(self, lag = 20):  
                """ Returns the rolling mean values as a new time-serie """
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                Values = self.securityC().Values()

                for i in (range(0,len(Dates))):
                        if ((i+lag)>=len(Dates)):
                                break
                        rocVal = (((Values[(i+lag)]/Values[i])*100)-100 )
                        mdates.append( Dates[i+lag] )
                        mvalues.append( rocVal )
                        
                return TimeSeries(dates = mdates, values = mvalues)

        # Exponential rolling mean (moving average) for MACD calculation.
        #This is different from the rExpMean in Time Serie class as it has a different smoothing constant
        def rExpMean(self, tscom, lag = 30):

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                K = 2./(lag+1)
                #K = 1./float(lag)
                notyet = True
        
                for i in range(0,len(Dates)):            
                    if notyet:
                        mean = Values[i]
                        notyet = False
                    else:
                        mean = mean-(K*(mean-Values[i]))
                        mdates.append( Dates[i] )
                        mvalues.append( mean )

                return TimeSeries(dates = mdates, values = mvalues)

        # Smoothing for ADX calc
        def rSmoothed(self, tscom, lag = 30):

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                        
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                K = float(lag-1)/lag
                #K = 1./float(lag)
                notyet = True
        
                for i in range(0,len(Dates)):            
                    if notyet:
                        mean = Values[i]
                        notyet = False
                    else:
                        mean = (K*mean)+Values[i]
                        mdates.append( Dates[i] )
                        mvalues.append( mean )

                return TimeSeries(dates = mdates, values = mvalues)

        # rExpmean for ADX calc
        def rExpMeanForADX(self, tscom, lag = 30):

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                K = float(lag-1)/lag
                #K = 1./float(lag)
                notyet = True
        
                for i in range(0,len(Dates)):            
                    if notyet:
                        mean = Values[i]
                        notyet = False
                    else:
                        mean = ((1-K)*Values[i])+(K*mean)
                        mdates.append( Dates[i] )
                        mvalues.append( mean )

                return TimeSeries(dates= mdates, values = mvalues)
                
        # Velocity for MACD (difference between exp. moving averages)
        
        def rVelocity(self, fastMA = 13, slowMA = 26): 
                return self.rExpMean(self.securityC(), fastMA)-self.rExpMean(self.securityC(), slowMA)
    
        # Moving Average Convergence/Divergence signal line to monitor crossovers
        
        def rMACDSig(self, lag = 9): 
                return self.rExpMean(self.rVelocity(), lag)

        #compare values in two time series xover is to check whether there was a recent crossover
        #Faster timeseries is tscom1, slower is tscom2
        #Return 1 as Timeseries value if ts1 element > ts2 element, -1 if ts1 < ts 2 otherwise return 0
        def rCompare(self, tscom1, tscom2, xover=False):

                if USECOM:
                        ts1 =win32com.client.Dispatch( tscom1 )
                        ts2 =win32com.client.Dispatch( tscom2 )
                else:
                        ts1 = tscom1
                        ts2 = tscom2

                data=[]
                size = min(len(ts1.Values()), len(ts2.Values()))
                mvalues=ts1.Dates()[-size:]
                lastu = ts1.Values()[-size]
                lastv = ts2.Values()[-size]
                for u,v in zip(ts1.Values()[-size:],ts2.Values()[-size:]):
                        if (((xover==True) & (lastu < lastv) & (u>v)) or ((xover==False) & (u>v))):
                                data.append(1)
                        elif (((xover==True) & (lastu > lastv) & (u<v)) or ((xover==False) & (u<v))):
                                data.append(-1)
                        else:
                                data.append(0)
                        lastu = u
                        lastv = v

                return TimeSeries(dates = mvalues, values = data)

        #compare values in two time series xover is to check whether there was a recent crossover
        #Faster timeseries is tscom1, slower is tscom2. The compare is done depending on whether market is above or below the level
        #Return 1 as Timeseries value if ts1 element > ts2 element, -1 if ts1 < ts 2 otherwise return 0
        def rCompareAboveBelow(self, tscom1, tscom2, xover=False, level = 0):

                if USECOM:
                        ts1 =win32com.client.Dispatch( tscom1 )
                        ts2 =win32com.client.Dispatch( tscom2 )
                else:
                        ts1 = tscom1
                        ts2 = tscom2

                data=[]
                size = min(len(ts1.Values()), len(ts2.Values()))
                mvalues=ts1.Dates()[-size:]
                lastu = ts1.Values()[-size]
                lastv = ts2.Values()[-size]
                for u,v in zip(ts1.Values()[-size:],ts2.Values()[-size:]):
                        if (((xover==True) & (lastu < lastv) & (u>v) & (v<level)) or ((xover==False) and (u>v))):
                                data.append(1)
                        elif (((xover==True) & (lastu > lastv) & (u<v) & (v>level)) or ((xover==False) and (u<v))):
                                data.append(-1)
                        else:
                                data.append(0)
                        lastu = u
                        lastv = v

                return TimeSeries(datess = mvalues, values = data)
                

        # This code assumes the three TS have the same dates 
        
        def TR(self): 
                spotYesterdayClose = Ether.Creator('TimeSerie', Dates=self.securityC().Dates()[1:], values=self.securityC().Values()[0:-1])

                tr1vals = (self.securityH() - self.securityL()).Values()[1:]
                tr2vals = (self.securityH() - spotYesterdayClose).Values()
                tr3vals = (spotYesterdayClose - self.securityL()).Values()

                TRDates = spotYesterdayClose.Dates()
                
                TRvals = [max([abs(tr1),abs(tr2),abs(tr3)]) for tr1,tr2,tr3 in zip(tr1vals,tr2vals,tr3vals)]
                
                return TimeSeries(dates=TRDates, values=TRvals)
        
    
        # Upper Bollinger Band
        
        def rBolliUpper(self, tscom, lag = 20, stdevband = 2):  
                """ Returns the upper Bollinger band values as a new time-serie """

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                
                def auxbollupper(ts, start, end):
                    length = end-start+1
                    if (length > 0):
                        return Dates[end],((sum(Values[start:end+1]))/length) + stdevband*(std(Values[start:end+1]))
                
                for i in (range(0,len(Dates))):
                        if ((i+lag)>len(Dates)):
                                break
                        dt, val = auxbollupper(ts, i, (i+lag-1))
                        mdates.append( dt )
                        mvalues.append( val )
                        
                return TimeSeries(dates = mdates, values = mvalues)


        # Lower Bollinger Band
        
        def rBolliLower(self, tscom, lag = 20, stdevband = 2):  
                """ Returns the lower Bollinger band values as a new time-serie """

                if USECOM:
                        ts =win32com.client.Dispatch( tscom )
                else:
                        ts = tscom
                
                mdates = []
                mvalues = []

                Dates = [ConvertDate(date) for date in ts.Dates()]
                Values = ts.Values()

                
                def auxbolllower(ts, start, end):
                    length = end-start+1
                    if (length > 0):
                        return Dates[end],((sum(Values[start:end+1]))/length) - stdevband*(std(Values[start:end+1]))
                
                for i in (range(0,len(Dates))):
                        if ((i+lag)>len(Dates)):
                                break
                        dt, val = auxbolllower(ts, i, (i+lag-1))
                        mdates.append( dt )
                        mvalues.append( val )
                        
                return TimeSeries(dates = mdates, values = mvalues)

        def rBollidiff(self, lag1 = 20, lag2=20, stdevband1 = 2,stdevband2=2):
                return self.rBolliupper(self.securityC(),lag1,stdevband1) - self.rBollilower(self.securityC(),lag2,stdevband2)
            
        # This code assumes the three TS have the same dates 
        
        def rADX(self, lag = 7):

                #print 'in adx'
                
                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                
                TrueRange = []
                PDM = []
                MDM = []
                STR = []
                SPDM = []
                SMDM = []
                PDI = []
                MDI = []
                DX = []
                ADXValues = []
                ADXDates = []

                LoValues = self.securityL().Values()
                HiValues = self.securityH().Values()
                ClValues = self.securityC().Values()
                

                #print ' entering for'
                for i in range(1,len(self.securityC().Dates())):
                    hiValue = HiValues[i]
                    loValue = LoValues[i]
                    truerange = max(abs(hiValue- loValue),abs(hiValue- ClValues[i-1]),abs(ClValues[i-1]- loValue))
                    TrueRange.append(truerange)
                

                    if  (hiValue - HiValues[i-1]) < (LoValues[i-1]- loValue):
                        pdm = 0
                        
                    else:
                        if (hiValue < HiValues[i-1]) & (LoValues[i-1] < loValue):
                            pdm = 0
                            
                        else:
                            pdm = hiValue -HiValues[i-1]
                    PDM.append(pdm)
                    

                    if  (LoValues[i-1] -loValue) < (hiValue- HiValues[i-1]):
                        mdm = 0
                        
                    else:
                        if (LoValues[i-1] < loValue) & (hiValue < HiValues[i-1]):
                            mdm = 0
                            
                        else:
                            mdm = LoValues[i-1] -loValue
                    MDM.append(mdm)
                #print 'ended for'
                
               

                strr = (sum(TrueRange[0:lag]))
                STR.append(strr)

                spdm = (sum(PDM[0:lag]))
                SPDM.append(spdm)

                smdm = (sum(MDM[0:lag]))
                SMDM.append(smdm)
                
                
                #print 'secdond for'
                for i in range(1,len(self.securityC().Dates())-lag):
                        
                        strr = ((float(lag-1)/(lag))*strr)+TrueRange[lag+i-1]
                        STR.append(strr)

                    
                        spdm = ((float(lag-1)/(lag))*spdm)+PDM[lag+i-1]
                        SPDM.append(spdm)

                   
                        smdm = ((float(lag-1)/(lag))*smdm)+MDM[lag+i-1]
                        SMDM.append(smdm)
                #print 'ended for'

                
                for i in range(0,len(self.securityC().Dates())-lag):   
                
                    pdi = (float(SPDM[i])/STR[i])
                    mdi = (float(SMDM[i])/STR[i])

                    PDI.append(pdi)
                    MDI.append(mdi)

                    dx = 100*abs((pdi-mdi)/(pdi+mdi))
                    DX.append(dx)
                #print 'ended third for'
                
                for i in range(0,len(DX)-lag+1):
                    if i== 0:
                        adx = (sum(DX[i:i+lag]))/lag
                        

                    else:
                      
                        adx = (DX[i+lag-1] + ((float(lag-1))*adx))/(lag)
                    ADXValues.append(adx)

                    dates = Dates[i + (2*lag) - 1]
                    ADXDates.append(dates)
                #print 'ended fourth for'
                               
                return TimeSeries(dates=ADXDates, values=ADXValues)
                
                #except IndexError:
                        #print 'Dates and values are possibly out of sync'



        #To determine MA crossover
        
        def rMaPositioning(self, fastMaVal, slowMaVal):
                fastMA = self.rMean(self.securityC(),min(fastMaVal, slowMaVal))
                slowMA = self.rMean(self.securityC(),max(fastMaVal, slowMaVal))
                maPosition = self.rCompare(fastMA, slowMA)
                return maPosition
                
        # definition of uptrend using ADX, regressionSlope, rMeanCross
        
        def isUptrend(self):   
                data=[]
                adxTS = self.rADX(7)
                slopeTS = self.rRegSlope(self.securityC(), 5)

                adxLevel = TimeSeries(dates=self.securityC().Dates(), values=[20.]*len(self.securityC().Values()))
                
                rSlopeLevel = TimeSeries(dates= self.securityC().Dates(), values=[0.]*len(self.securityC().Values()))

                maPosition = self.rMaPositioning(5, 21).Values()
                
                adx = self.rCompare(adxTS, adxLevel).Values()
                slope = self.rCompare(slopeTS, rSlopeLevel).Values()

                size = min(len(maPosition), len(adx), len(slope))
                mvalues = self.securityC().Dates()[-size:]
                
                for u,v,w in zip(maPosition[-size:], adx[-size:], slope[-size:]):
                        if (u==v==w==1):
                                data.append(1) #A trending phase
                        else:
                                data.append(0)  #Not a trending phase
                return TimeSeries(dates= mvalues, values = data)

                
        # definition of downtrend using ADX, regressionSlope, rMeanCross
        
        def isDowntrend(self): 

                data=[]
                
                adxTS = self.rADX(7)
                slopeTS = self.rRegSlope(self.securityC(), 5)
            
                adxLevel = TimeSeries(dates= self.securityC().Dates(), values=[20.]*len(adxTS.Values())) #20
                rSlopeLevel = TimeSeries(dates= self.securityC().Dates(), values=[0.]*len(slopeTS.Values())) #0
                
                maPosition = self.rMaPositioning(5, 21).Values()
                adx = self.rCompare(adxTS, adxLevel).Values()
                slope = self.rCompare(slopeTS, rSlopeLevel).Values()
                
                
                size = min(len(maPosition), len(adx), len(slope))
                mvalues = self.securityC().Dates()[-size:]
                
                for u,v,w in zip(maPosition[-size:], adx[-size:], slope[-size:]):
                        if ((u==-1) & (w==-1) & (v==1)):
                                data.append(1)  #A trending phase
                        else:
                                data.append(0)  #Not a trending phase
                return TimeSeries(dates= mvalues, values = data)

        #Return timeseries to compare
        def getFlatLine(self, val1=0):
            compareSeries = TimeSeries(dates=self.securityC().Dates(), values=[float(val1)]*len(self.securityC().Values()))
            return compareSeries

        def compareSeries(self, method, val1, val2):
            compareSeries = None
            if (val2 == 0):
                compareSeries = self.securityC()
            else:
                compareSeries = method(self.securityC(), val2)
            return compareSeries
        
        #Returns a timeseries object with dates of entry triggers and index location as value
        
        def entrySignals(self, ttype = 'Buy', methodName = 'rRsi', val1=9, val2=20, xOver = False):
            data=[]
            mvalues=[]

            method = getattr(self, methodName)
            
            
            if (methodName == 'rStoD') and (xOver == True): #Entry condition for Stochastics
                oscillator = method(val1)
                toCompare = self.rStoK(val1)
                entries = self.rCompareAboveBelow(toCompare, oscillator, xOver, val2)
            else:
                if (methodName == 'Engulf') or (methodName == 'Doji') or (methodName == 'KRD'):
                    finalMethod = method(methodName, ttype)
                    return finalMethod()
                elif (methodName == 'hammer') or (methodName == 'shootingStar') or (methodName == 'bullDoji') or (methodName == 'bullKRD'):
                    return method()
                elif (methodName == 'rMean'):
                    oscillator = method(self.securityC(),val1)
                    toCompare = self.compareSeries(method, val1, val2)
                elif (methodName == 'rStoD'):
                    oscillator = method(val1)
                    toCompare = self.rStoK(val1)
                elif (methodName == 'rRsi') or (methodName == 'rADX'):
                    oscillator = self.getFlatLine(val2)
                    toCompare = method(val1)
              
                #If using maxHist then avoid taking all the signals due to 'lag factor', thus the 'Legit Date'
                #dateAfter = techbacktest.addTenorToDate(startup.Startup.firstDate, period)
                entries = self.rCompare(toCompare, oscillator, xOver)
                
                

            Dates = [ConvertDate(date) for date in self.securityC().Dates()]
            entryDates = [ConvertDate(date) for date in entries.Dates()]
           
            entryValues = entries.Values()
            
            
            for d in (entryDates):
                #if (d<dateAfter):
                #continue
                ind = entryDates.index(d)
                val = entryValues[ind]
                if (val == 1 and ttype == 'Buy'):
                    dateIndex = Dates.index(d)
                    data.append(dateIndex)
                    mvalues.append(d)
                elif (val == -1 and ttype == 'Sell'):
                    dateIndex = Dates.index(d)
                    data.append(dateIndex)
                    mvalues.append(d)

            
                        
            return Ether.Creator('TimeSerie', Dates = mvalues, values = data)

        def Engulf(self, methodName, ttype):
            if ttype == 'Buy':
                return getattr(self, "bull"+methodName)
            else:
                return getattr(self, "bear"+methodName)

        def Doji(self, methodName, ttype):
            if ttype == 'Buy':
                return getattr(self, "bull"+methodName)
            else:
                return getattr(self, "bear"+methodName)

        def KRD(self, methodName, ttype):
            if ttype == 'Buy':
                return getattr(self, "bull"+methodName)
            else:
                return getattr(self, "bear"+methodName)

        
            
        # Preceding downtrend is needed for this to be valid
        #Close < Min(Open, Close (previous bar)) AND Open > Max(Open, Close (previous bar))
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bearEngulf(self, lag = 30000):
                data=[]
                mvalues=[]
                
                notyet = True
                lastC = 0
                lastO = 0

                tr=self.isUptrend().Values()
                if (lag==0):
                        return Ether.Creator('TimeSerie', Dates = [], values = [])
                else:
                        size=len(tr)
                        
                op=self.securityO().Values()[-size:]
                cl=self.securityC().Values()[-size:]
                tr=tr[-size:]  

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                 
                for o,c,t,d in zip(op, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            if ((c<min(lastO,lastC)) & (o>max(lastO, lastC)) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastC = c
                        lastO = o
                return TimeSeries(dates= mvalues, values = data)

        # Preceding uptrend is needed for this to be valid 
        #Close > Max(Open, Close (previous bar)) AND Open < Min(Open, Close (previous bar))
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bullEngulf(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastC = 0
                lastO = 0

                tr=self.isDowntrend().Values()
                if (lag==0):
                        return Ether.Creator('TimeSerie', Dates = [], values = [])
                else:
                        size=len(tr)

                op=self.securityO().Values()[-size:]
                cl=self.securityC().Values()[-size:]
                tr=tr[-size:]  

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                    
                for o,c,t,d in zip(op, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            if ((c>max(lastO,lastC)) & (o<min(lastO, lastC)) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastC = c
                        lastO = o
                return TimeSeries(dates= mvalues, values = data)

        # Preceding uptrend is needed for this to be valid 
        #In an uptrend: Close < Low (previous bar) AND High > High (previous bar)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bearKRD(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastL = 0
                lastH = 0

                tr=self.isUptrend().Values()
                if (lag==0):
                        return Ether.Creator('TimeSerie', Dates = [], values = [])
                else:
                        size=len(tr)
                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]
                tr=tr[-size:]

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                
                for h,l,c,t,d in zip(hi, lo, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            if ((c<lastL) & (h>lastH) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastL = l
                        lastH = h
                return TimeSeries(dates = mvalues, values = data)
              

        # Preceding downtrend is needed for this to be valid
        #In a downtrend: Close > High (previous bar) AND Low < Low (previous bar)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bullKRD(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastL = 0
                lastH = 0

                
                tr=self.isDowntrend().Values()
                
                
                if (lag==0):
                        return TimeSeries(dates= [], values = [])
                else:
                        size=len(tr)

                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]
                tr=tr[-size:]

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                
                for h,l,c,t,d in zip(hi, lo, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            if ((c>lastH) & (l<lastL) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastL = l
                        lastH = h
                return TimeSeries(dates = mvalues, values = data)



        # Preceding uptrend is needed for this to be valid 
        #ABS (Open – Close) < (0.01% * Open) AND Bar posts a new two day high (in an uptrend)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bearDoji(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastH = 0

                tr=self.isUptrend().Values()
                dt = self.isUptrend().Dates()
                
                if (lag==0):
                        return TimeSeries(dates= [], values = [])
                else:
                        size=len(tr)

                #print 'tr', len(tr), len(self.securityO.Values()), len(self.securityH.Values()), len(self.securityL.Values()), len(self.securityC.Values())

                tr=tr[-size:]                
                op=self.securityO().Values()[-size:]
                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]

                #for d,o,h,l,c in zip(dt, op, hi, lo, cl):
                        #print d, o, h, l, c
                
                dates=Dates[-size:]

                for o,h,l,c,t,d in zip(op, hi, lo, cl, tr, dates):
                       
                        if notyet:
                                notyet = False
                        else:
                            rnge=abs(h-l)
                            #print ((abs(o-c))<=((0.00001)*o)), (h>lastH), (abs(o-c) < (rnge/4)), (t==1), d
                            if (((abs(o-c))<=((0.00001)*o)) & (h>lastH) & (abs(o-c) < (rnge/4)) & (t==1)):
                                
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastH = h
                return TimeSeries(dates = mvalues, values = data)



        # Preceding downtrend is needed for this to be valid
        #ABS (Open – Close) < (0.01% * Open) AND ABS (Open – Close) < Range/4 AND Bar posts a new two day low (in a downtrend)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def bullDoji(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastL = 0

                tr=self.isDowntrend().Values()
                if (lag==0):
                        return TimeSeries(dates = [], values = [])
                else:
                        size=len(tr)
                
                tr=tr[-size:]                
                op=self.securityO().Values()[-size:]
                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]                

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                
                for o,h,l,c,t,d in zip(op, hi, lo, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            rnge=abs(h-l)
                            if ((abs(o-c) <= (0.00001)*o) & (l<lastL) & (abs(o-c) < (rnge/4)) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastL = l
                return TimeSeries(dates= mvalues, values = data)


        # Preceding downtrend is needed for this to be valid
        #ABS (Open – Close) < Range/2.5 AND High-Max(Close, Open)<= Range/30 AND Bar posts a new two day low (in a downtrend)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def hammer(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastL = 0

                tr=self.isDowntrend().Values()
                if (lag==0):
                        return TimeSeries(dates = [], values = [])
                else:
                        size=len(tr)
                
                tr=tr[-size:]                
                op=self.securityO().Values()[-size:]
                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]

                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                
                for o,h,l,c,t,d in zip(op, hi, lo, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            rnge=abs(h-l)
                            if ((abs(o-c) <= (rnge/2.5)) & (l<lastL) & ((h-(max(c,o)))<=(rnge/30)) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                            lastL = l
                return TimeSeries(dates = mvalues, values = data)


        # Preceding uptrend is needed for this to be valid 
        #ABS (Open – Close) < Range/2.5 AND Min(Close, Open) - Low <= Range/30 AND Bar posts a new two day high (in a downtrend)
        #Returns a timeseries object with dates of triggers and index location as value
        
        def shootingStar(self, lag = 30000): 
                data=[]
                mvalues=[]
                
                notyet = True
                lastH = 0

                tr=self.isUptrend().Values()
                if (lag==0):
                        return TimeSeries(dates = [], values = [])
                else:
                        size=len(tr)
                
                tr=tr[-size:]                
                op=self.securityO().Values()[-size:]
                hi=self.securityH().Values()[-size:]
                lo=self.securityL().Values()[-size:]
                cl=self.securityC().Values()[-size:]
                
                Dates = [ConvertDate(date) for date in self.securityC().Dates()]
                dates=Dates[-size:]
                
                for o,h,l,c,t,d in zip(op, hi, lo, cl, tr, dates):
                        if notyet:
                                notyet = False
                        else:
                            rnge=abs(h-l)
                            if ((abs(o-c) <= (rnge/2.5)) & (h>lastH) & ((min(c,o)-l)<=(rnge/30)) & (t==1)):
                                ind = Dates.index(d)
                                length = len(Dates)
                                if ((length-ind)<=lag):
                                        data.append(ind)
                                        mvalues.append(d)
                        lastH = h
                return TimeSeries(dates = mvalues, values = data)

           
                
        """
        def readFile(self): #Reading from file for testing purposes
                f=open('usdjpy.txt', 'r')
                dates=[]
                op=[]
                high=[]
                low=[]
                close=[]
                while 1:
                        line=f.readline()
                        if not line:
                                break
                        parts=line.split('\t')
                        d=parts[0].split('/')
                        dates.append(datetime.date(int(d[2]), int(d[1]), int(d[0])))
                        op.append(float(parts[1]))
                        high.append(float(parts[2]))
                        low.append(float(parts[3]))
                        close.append(float(parts[4][:-1]))
                self.securityO=Ether.Creator('TimeSerie', Dates = dates, Values = op)
                self.securityH=Ether.Creator('TimeSerie', Dates = dates, Values = high)
                self.securityL=Ether.Creator('TimeSerie', Dates = dates, Values = low)
                self.securityC=Ether.Creator('TimeSerie', Dates = dates, Values = close)
        """



##Instantiation example
#print 'start', datetime.datetime.today()
#t=TechLib('EUR','GBP',360)
#t.readFile()
#t.convertToWeekly(t.securityC, 'C')
#print t.rMean(t.securityC)
#t.rMean(t.securityC)
#t.maPositioning(21,5)
#t.bull('rADX')
#t.bull('rRsi')
#t.bear('rRsi')
#t.bull('rVelocity')
#t.bear('rVelocity')
#t.bull('rStoD')
#print t.securityC.Values(), t.securityC.Dates()
#print 'done', datetime.datetime.today()


#t.readFile()
