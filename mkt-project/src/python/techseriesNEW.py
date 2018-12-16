# -*- coding: cp1252 -*-
import sys
import os
import datetime,time
import math
from numpy import median
import array
from techlib import TechLib
from FileReaderNEW import Filereader
from timeseries import TimeSeries
import logging

if not 'LOG_DIR' in os.environ:
	print 'ERROR: LOG_DIR is not an environment variable.'
	sys.exit(2)

LOG_FILENAME = '%s/techseries-%s.log' %(os.environ['LOG_DIR'], str(datetime.date.today()))

logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

# Wrapper class that includes Technical Analysis functions, analyses them, and returns the state of them
# When comparing TimeSerie objects the code assumes that the O, H, L, C vectors are aligned in dates and values
# Author: DS

#import win32com.client
#import pythoncom

#maxHist = 250
today = datetime.date.today()
firstLegitDate = datetime.date(1992,5,18)
maxHist = (today - firstLegitDate).days

dicoTS = {}

def ConvertDate(d):
	return d


def acTechSeries(asset='EUR', base='USD', horizon=maxHist, period='D', fileFlag='True'):#If fileFlag = False, read data from Asset Control. If fileFlag = True, read data from .txt file, with date in UK format in column 1(dd/mm/yyyy), then O,H,L,C in columns 2-5. If fileFlag = 'Pickle', read data from previously pickled file. 
        
        try:
                ts = dicoTS[(asset,base,horizon,period)]
                #logx.Write('Found this TS in the dico ;'+asset+','+base+','+str(horizon)+','+period)
                return ts
        except:
                ts = TechSeries(asset, base, horizon, period, fileFlag)
                #print 'Ether created'
                dicoTS[(asset,base,horizon,period)] = ts
                return ts

                
class TechSeries(TechLib):

        def __init__(self, asset, base, horizon, period, fileFlag):
                self.asset = asset
                self.base = base
                self.horizon = horizon
                self.period = period
                self.fileFlag = fileFlag
		self.todays = True
		
	def IsTodays(self):
		return self.todays
                
        def Asset(self):
                return self.asset

        def Base(self):
                return self.base

        def Horizon(self):
                return self.horizon

        def Period(self):
                return self.period
        
        def Flag(self):
                return self.fileFlag
        
        
        # Constructor - also converts daily data to weekly
        def RawData(self):
                asset = self.Asset()
                base = self.Base()
                horizon = self.Horizon()
                period = str(self.Period())
                flag = self.Flag()
                cacheRetrieved = False
                commondates = []
                if (horizon > maxHist):
                        raise ValueError('horizon too large')
                if (horizon<maxHist):
                        #print 'Trying to create from internal cache...'
                        #logx.Write('Trying to create series from internal cache...')
                        maxTS = acTechSeries(asset,base,maxHist,period, flag)
                        securityO = TimeSeries(dates = maxTS.securityO().Dates()[-horizon:], values = maxTS.securityO().Values()[-horizon:])                            
                        securityH = TimeSeries(dates = maxTS.securityH().Dates()[-horizon:], values = maxTS.securityH().Values()[-horizon:])  
                        securityL = TimeSeries(dates = maxTS.securityL().Dates()[-horizon:], values = maxTS.securityL().Values()[-horizon:])  
                        securityC = TimeSeries(dates = maxTS.securityC().Dates()[-horizon:], values = maxTS.securityC().Values()[-horizon:])
                        cacheRetrieved = True 
                else:
                        if (period == 'W'): # At this point, maxHist for 'W' period is not there
                                #print 'Creating daily series from internal cache to generate weekly series...'
                                maxTS = acTechSeries(asset,base,maxHist,'D',flag) #This MUST be loaded up in the startup
                                securityC = maxTS.securityC()
                                securityH = maxTS.securityH()
                                securityL = maxTS.securityL()
                                securityO = maxTS.securityO()
                        else:
                                #print 'Creating daily series from spotTS instead of dicoTS'
                                #logx.Write('Creating from scratch ; '+str(horizon)+','+str(maxHist))
                                # Retrieving any data for the very first time below
                                fileReader = Filereader()
                                
                                if (self.Flag()) == 'True':
                                        securityO, securityH, securityL, securityC = fileReader.createTStxt(asset=asset,base=base,priceflag='All')
                                elif self.Flag() == 'Pickle':
                                        securityO, securityH, securityL, securityC = fileReader.createTSpickle(asset=asset,base=base)
                                        #print 'received timeserie', datetime.datetime.today()
                                else:
                                        if (base != ''): #Assuming an FX security
                                                from fxstdtimeserie import spotTS
                                                securityC, securityH, securityL, securityO = spotTS(asset, base, horizon, ['dqpp_close','dqpp_high','dqpp_low','dqpp_open'])
                                        else: #non FX security
                                                securityC, securityH, securityL, securityO = acTS(asset, horizon)
                                
                                #Filter dates to include only rows that have all -- O,H,L and C

                                
                if (len(commondates) == 0): #If no filtering was done, get from what is existing
                        commondates =[ConvertDate(date) for date in securityC.Dates()]
                        
                if ((period == 'W') and not(cacheRetrieved)): #Convert daily data to weekly data
                        mdates = []
                        cvalues = []
                        ovalues = []
                        hvalues = []
                        lvalues = []
                        

                        #Find the first monday
                        for nextDate in commondates:
                                if (days.DayOfWeek(nextDate) == 'monday'):
                                        break

                        while nextDate <= commondates[-1]: # MZ : if thing it is possible for this to have a week from monday to monday because of the way the last day is rolled back
                                        startDate = nextDate
                                        indStart = ''
                                        indEnd = ''
                                        for k in range(5):#(0,7):
                                                try:
                                                        indStart = commondates.index(nextDate)
                                                        break
                                                except:
                                                        nextDate = days.NextWeekDay(nextDate) #nextDate+datetime.timedelta(1)
                                                        
                                        endDate = startDate+datetime.timedelta(5)
                                        nextDate = startDate+datetime.timedelta(7)
                                        if (indStart == ''):
                                                continue

                                        for k in range(0,7):
                                                try:
                                                        indEnd = commondates.index(endDate)
                                                        break
                                                except:
                                                        endDate = endDate-datetime.timedelta(1) # MZ : see comment above
                                                        
                                        if (indEnd == ''):
                                                continue
                                        
                                        mdates.append(endDate)
                                        ovalues.append(securityO.Values()[indStart])
                                        minVal = min(securityL.Values()[indStart:(indEnd+1)])
                                        lvalues.append(minVal)
                                        maxVal = max(securityH.Values()[indStart:(indEnd+1)])
                                        hvalues.append(maxVal)
                                        cvalues.append(securityC.Values()[indEnd])
                
                        securityO = TimeSeries(dates = mdates, values = ovalues)
                        securityH = TimeSeries(dates =mdates, values = hvalues)
                        securityL = TimeSeries(dates =mdates, values = lvalues)
                        securityC = TimeSeries(dates = mdates, values = cvalues)

                
		if securityO.dates[-1] != datetime.date.today():
			self.todays = False
			logging.warning('WARNING: Running on stale date. The last date in your database is %s' %securityO.dates[-1])
                return [securityO, securityH, securityL, securityC]


        def securityO(self):
                return self.RawData()[0]

        def OpenValues(self):
                return list(self.securityO().Values())

        def securityH(self):
                return self.RawData()[1]

        def HighValues(self):
                return list(self.securityH().Values())

        def securityL(self):
                return self.RawData()[2]

        def LowValues(self):
                return list(self.securityL().Values())

        def securityC(self):
                return self.RawData()[3]

        def CloseValues(self):
                return list(self.securityC().Values())

        def getOHLCIndices(self, closeDate):
                try:
                        closeIndex = self.securityC().Dates().index(closeDate)
                        openIndex = self.securityO().Dates().index(closeDate)
                        highIndex = self.securityH().Dates().index(closeDate)
                        lowIndex = self.securityL().Dates().index(closeDate)
                        return openIndex, highIndex, lowIndex, closeIndex
                except:
                        return -1,-1,-1,-1
                
        def currentMean(self, lag = 21, nbBizDaysAgo = 0 ):
                if (nbBizDaysAgo == 0):
                        dt, val = self.mean(self.securityC(), (len(self.securityC().Dates())-lag), (len(self.securityC().Dates())-1))
                else:
                        dt, val = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-lag), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
# 		print 'securityC', self.securityC()
# 		print 'date', dt
# 		print 'value', val
# 		print 'nbBizDaysAgo', nbBizDaysAgo

                returnDate = ''
                returnVal = ''
                if (dt != ''):
                        returnDate = ConvertDate(dt).isoformat()
                        if (val<self.securityC().Values()[-nbBizDaysAgo-1]):
                                returnVal = 'Bullish'
                        elif (val>self.securityC().Values()[-nbBizDaysAgo-1]):
                                returnVal = 'Bearish'
                        else:
                                returnVal = 'Neutral'
                return [returnDate, returnVal]

        def maCrossOver(self, fastMaVal, slowMaVal, nbBizDaysAgo = 0):
                returnVal = ''
                returnDate = ''
                if (nbBizDaysAgo == 0):
                        dtFast, valFast = self.mean(self.securityC(), (len(self.securityC().Dates())-fastMaVal), (len(self.securityC().Dates())-1))
                        dtSlow, valSlow = self.mean(self.securityC(), (len(self.securityC().Dates())-slowMaVal), (len(self.securityC().Dates())-1))
                else:
                        dtFast, valFast = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-fastMaVal), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
                        dtSlow, valSlow = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-slowMaVal), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
                
                if (dtFast != ''):
                        returnDate = ConvertDate(dtFast).isoformat()
                        if valFast>valSlow:
                                returnVal = 'Bullish'
                        elif valFast<valSlow:
                                returnVal = 'Bearish'
                        else:
                                returnVal = 'Neutral'
                return [returnDate, returnVal]

        def currentBolli(self, lag1=20,lag2=0):
                returnVal = ''
                returnDate = ''
                tsslow = self.rMean(self.securityC(),lag=lag1)
                tsfast = self.rMean(self.securityC(),lag=lag2+1)
                tsupper = self.rBolliupper(self.securityC(),lag=lag1)
                tslower = self.rBollilower(self.securityC(),lag=lag1)
                
                if (len(tsslow.Dates())>0):
                        returnDate = ConvertDate(tsfast.Dates()[-1]).isoformat()
                        if tsfast.Values()[-1]>tsslow.Values()[lag2-lag1]:
                               if tsfast.Values()[-1] < tsupper.Values()[lag2-lag1]:
                                        returnVal = 'Bullish'
                        elif tsfast.Values()[-1]< tsslow.Values()[lag2-lag1]:
                               if tsfast.Values()[-1] > tslower.Values()[lag2-lag1]:
                                        returnVal = 'Bearish'
                        else:
                                returnVal = 'Neutral'
                return [returnDate, returnVal]

        def currentADX(self, lag = 7, level = 25, nbBizDaysAgo = 0):
                ts = self.rADX(lag)
                #print ts.Values(), 'ADX'
                returnDate = ''
                returnVal = ''
                if (len(ts.Dates())>0):
                        returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                        if ts.Values()[-nbBizDaysAgo-1]>level:
                                returnVal = 'Trending'
                        else:
                                returnVal = 'Ranging' 
                return [returnDate, returnVal]
        
        def currentMACD(self, fastMaVal = 13, slowMaVal = 26, nbBizDaysAgo = 0):
                ts = self.rVelocity(fastMaVal, slowMaVal)
                #print ts.Values(), 'MACD'
                returnDate = ''
                returnVal = ''
                tsSig = self.rMACDSig(9) # Defaulted to 9 period signal
                if ((len(ts.Dates())>0) and (len(tsSig.Dates())>0)):
                        #print 'macd',ts.Values()[-1]
                        returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                        if ((ts.Values()[-nbBizDaysAgo-1]>0) and (ts.Values()[-nbBizDaysAgo-1]<tsSig.Values()[-nbBizDaysAgo-1])):
                                returnVal = 'Neutral'
                        elif ((ts.Values()[-nbBizDaysAgo-1]>0) and (ts.Values()[-nbBizDaysAgo-1]>tsSig.Values()[-nbBizDaysAgo-1])):
                                returnVal = 'Bullish'
                        elif ((ts.Values()[-nbBizDaysAgo-1]<0) and (ts.Values()[-nbBizDaysAgo-1]>tsSig.Values()[-nbBizDaysAgo-1])):
                                returnVal = 'Neutral'
                        elif ((ts.Values()[-nbBizDaysAgo-1]<0) and (ts.Values()[-nbBizDaysAgo-1]<tsSig.Values()[-nbBizDaysAgo-1])):
                                returnVal = 'Bearish'
                        else:
                                returnVal = ''
                        #return {'Dates':ConvertDate(ts.Dates()[-1]).isoformat(), 'Values':val}

                return [returnDate, returnVal]

        def currentRSI(self, period = 9, obLevel = 80, osLevel = 20, nbBizDaysAgo = 0):
                ts = self.rRsi(period)
                returnDate = ''
                returnVal = '' 
                if (len(ts.Dates())>0):
                        returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                        if ts.Values()[-nbBizDaysAgo-1]>obLevel:
                                returnVal = 'Overbought'
                        elif ts.Values()[-nbBizDaysAgo-1]<osLevel:
                                returnVal = 'Oversold'
                        else:
                                returnVal = 'Neutral'
                return [returnDate, returnVal]


        def currentStochastics(self, period = 10, obLevel = 80, osLevel = 20, nbBizDaysAgo = 0):
                
                tsD = self.rStoD(period)
                tsK = self.rStoK(period)
                returnDate = ''
                returnVal = ''
                if (len(tsD.Dates())>0):
                        returnDate = ConvertDate(tsD.Dates()[-nbBizDaysAgo-1]).isoformat()
                        if tsD.Values()[-nbBizDaysAgo-1]>obLevel:
                                returnVal = 'Overbought'
                        elif tsD.Values()[-nbBizDaysAgo-1]<osLevel:
                                returnVal = 'Oversold'
                        elif (tsD.Values()[-nbBizDaysAgo-1]<=obLevel) and (tsD.Values()[-nbBizDaysAgo-1]>=osLevel) and tsK.Values()[-nbBizDaysAgo-1]>tsD.Values()[-nbBizDaysAgo-1]:
                                returnVal = 'Bullish'
                        elif (tsD.Values()[-nbBizDaysAgo-1]<=obLevel) and (tsD.Values()[-nbBizDaysAgo-1]>=osLevel) and tsK.Values()[-nbBizDaysAgo-1]<tsD.Values()[-nbBizDaysAgo-1]:
                                returnVal = 'Bearish'
                        else:
                                returnVal = 'Neutral'
                return [returnDate, returnVal]

       
        def filterSignal(self, tsBulls, tsBears):
                bullLength = len(tsBulls.Dates())
                bearLength = len(tsBears.Dates())
                date = ''
                val = ''
                active = ''
                bullOrBear = ''

                Spots = list(self.securityC().Values())
                Highs = list(self.securityH().Values())
                Lows = list(self.securityL().Values())

                if (bullLength > 0):
                        if (bearLength>0):
                                if (tsBulls.Dates()[-1]>tsBears.Dates()[-1]):
                                        date = tsBulls.Dates()[-1]
                                        close,endPos = determineExit(Spots,Highs[tsBulls.Values()[-1]],Lows[tsBulls.Values()[-1]],'stoploss', tsBulls.Values()[-1],(len(Spots)-1),'Buy',tsBulls.Values()[-1])
                                        if (endPos == 0):
                                                active = 'Yes'
                                        else:
                                                active = 'No'
                                        bullOrBear = 'Bullish'
                                elif (tsBulls.Dates()[-1]<tsBears.Dates()[-1]):
                                        date = tsBears.Dates()[-1]
                                        close,endPos = determineExit(Spots,Highs[tsBears.Values()[-1]],Lows[tsBears.Values()[-1]],'stoploss', tsBears.Values()[-1],(len(Spots)-1),'Sell',tsBears.Values()[-1])
                                        if (endPos == 0):
                                                active = 'Yes'
                                        else:
                                                active = 'No'
                                        bullOrBear = 'Bearish'
                        else:
                                date = tsBulls.Dates()[-1]
                                close,endPos = determineExit(Spots,Highs[tsBulls.Values()[-1]],Lows[tsBulls.Values()[-1]],'stoploss', tsBulls.Values()[-1],(len(Spots)-1),'Buy',tsBulls.Values()[-1])
                                if (endPos == 0):
                                        active = 'Yes'
                                else:
                                        active = 'No'
                                bullOrBear = 'Bullish'

                elif (bearLength > 0):
                        date = tsBears.Dates()[-1]
                        close,endPos = determineExit(Spots,Highs[tsBears.Values()[-1]],Lows[tsBears.Values()[-1]],'stoploss', tsBears.Values()[-1],(len(Spots)-1),'Sell',tsBears.Values()[-1])
                        if (endPos == 0):
                                active = 'Yes'
                        else:
                                active = 'No'
                        bullOrBear = 'Bearish'
                if (date!=''):
                        return [ConvertDate(date).isoformat(),bullOrBear, active]
                else:
                        return ['','','']

                        
        def indicesWithin(self, ts, lag = 25, nbBizDaysAgo = 0):
                endIndex = nbBizDaysAgo - 1
                startIndex = endIndex - lag
                validPatterns = []
                validDates = []
                for index, date in zip(ts.Values(), ts.Dates()):
                        if (index >= startIndex) and (index <=endIndex):
                                validPatterns.append(index)
                                validDates.append(date)
                return TimeSeries(dates = validDates, values = validPatterns)

        def lastDoji(self, lag = 25, nbBizDaysAgo = 0):
                if (nbBizDaysAgo != 0):
                        tsBulls = self.indicesWithin(self.bullDoji(lag=maxHist), lag, nbBizDaysAgo)
                        tsBears = self.indicesWithin(self.bearDoji(lag=maxHist), lag, nbBizDaysAgo)                        
                else:
                        tsBulls = self.bullDoji(lag)
                        tsBears = self.bearDoji(lag)
                
                return self.filterSignal(tsBulls, tsBears)

        def lastKRD(self, lag = 25, nbBizDaysAgo = 0):
                if (nbBizDaysAgo != 0):
                        tsBulls = self.indicesWithin(self.bullKRD(lag=maxHist), lag, nbBizDaysAgo)
                        tsBears = self.indicesWithin(self.bearKRD(lag=maxHist), lag, nbBizDaysAgo)
                else:
                        tsBulls = self.bullKRD(lag)
                        tsBears = self.bearKRD(lag)
                return self.filterSignal(tsBulls, tsBears)

        def lastEngulf(self, lag = 25, nbBizDaysAgo = 0):
                if (nbBizDaysAgo != 0):
                        tsBears = self.indicesWithin(self.bearEngulf(lag=maxHist), lag, nbBizDaysAgo)
                        tsBulls = self.indicesWithin(self.bullEngulf(lag=maxHist), lag, nbBizDaysAgo)
                else:
                        tsBears = self.bearEngulf(lag)
                        tsBulls = self.bullEngulf(lag)
                return self.filterSignal(tsBulls, tsBears)

        def lastHammer(self, lag = 25, nbBizDaysAgo = 0):
                if (nbBizDaysAgo != 0):
                        tsBulls = self.indicesWithin(self.hammer(lag=maxHist), lag, nbBizDaysAgo)
                        tsBears = self.indicesWithin(self.shootingStar(lag=maxHist), lag, nbBizDaysAgo)
                else:
                        tsBulls = self.hammer(lag)
                        tsBears = self.shootingStar(lag)
                return self.filterSignal(tsBulls, tsBears)

        # NEW FUNCTIONS---------------------------------------------------------------------------------------------
        
        def allMean(self,lag = 21):
                
                returnValAll = []
                returnDateAll = []
                L = len(self.securityC().Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        if (nbBizDaysAgo == 0):
                                dt, val = self.mean(self.securityC(), (len(self.securityC().Dates())-lag), (len(self.securityC().Dates())-1))
                        else:
                                dt, val = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-lag), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
                        returnDate = ''
                        returnVal = ''
                        if (dt != ''):
                                returnDate = ConvertDate(dt).isoformat()
                                if (val<self.securityC().Values()[-nbBizDaysAgo-1]):
                                        returnVal = 'Bullish'
                                elif (val>self.securityC().Values()[-nbBizDaysAgo-1]):
                                        returnVal = 'Bearish'
                                else:
                                        returnVal = 'Neutral'
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                                        
                return [returnDateAll, returnValAll]

        
        def allmaCrossOver(self, fastMaVal, slowMaVal):

                returnValAll = []
                returnDateAll = []
                L = len(self.securityC().Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        returnVal = ''
                        returnDate = ''
                        if (nbBizDaysAgo == 0):
                                dtFast, valFast = self.mean(self.securityC(), (len(self.securityC().Dates())-fastMaVal), (len(self.securityC().Dates())-1))
                                dtSlow, valSlow = self.mean(self.securityC(), (len(self.securityC().Dates())-slowMaVal), (len(self.securityC().Dates())-1))
                        else:
                                dtFast, valFast = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-fastMaVal), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
                                dtSlow, valSlow = self.mean(self.securityC(), (len(self.securityC().Dates()[:-nbBizDaysAgo])-slowMaVal), (len(self.securityC().Dates()[:-nbBizDaysAgo])-1))
                
                        if (dtFast != ''):
                                returnDate = ConvertDate(dtFast).isoformat()
                                if valFast>valSlow:
                                        returnVal = 'Bullish'
                                elif valFast<valSlow:
                                        returnVal = 'Bearish'
                                else:
                                        returnVal = 'Neutral'
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                                           
                return [returnDateAll, returnValAll]

        
        def allADX(self, lag = 7, level = 25):
                ts = self.rADX(lag)
                #print ts.Values(), 'ADX'
                returnValAll = []
                returnDateAll = []
                L = len(ts.Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        returnDate = ''
                        returnVal = ''
                        if (len(ts.Dates())>0):
                                returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                                if ts.Values()[-nbBizDaysAgo-1]>level:
                                        returnVal = 'Trending'
                                else:
                                        returnVal = 'Ranging'
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                                           
                return [returnDateAll, returnValAll]

        
        def allMACD(self, fastMaVal = 13, slowMaVal = 26):
                ts = self.rVelocity(fastMaVal, slowMaVal)
                #print ts.Values(), 'MACD'
                tsSig = self.rMACDSig(9) # Defaulted to 9 period signal
                returnValAll = []
                returnDateAll = []
                L = len(tsSig.Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        returnDate = ''
                        returnVal = ''
                        
                        if ((len(ts.Dates())>0) and (len(tsSig.Dates())>0)):
                                #print 'macd',ts.Values()[-1]
                                returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                                if ((ts.Values()[-nbBizDaysAgo-1]>0) and (ts.Values()[-nbBizDaysAgo-1]<tsSig.Values()[-nbBizDaysAgo-1])):
                                        returnVal = 'Neutral'
                                elif ((ts.Values()[-nbBizDaysAgo-1]>0) and (ts.Values()[-nbBizDaysAgo-1]>tsSig.Values()[-nbBizDaysAgo-1])):
                                        returnVal = 'Bullish'
                                elif ((ts.Values()[-nbBizDaysAgo-1]<0) and (ts.Values()[-nbBizDaysAgo-1]>tsSig.Values()[-nbBizDaysAgo-1])):
                                        returnVal = 'Neutral'
                                elif ((ts.Values()[-nbBizDaysAgo-1]<0) and (ts.Values()[-nbBizDaysAgo-1]<tsSig.Values()[-nbBizDaysAgo-1])):
                                        returnVal = 'Bearish'
                                else:
                                        returnVal = ''
                                #return {'Dates':ConvertDate(ts.Dates()[-1]).isoformat(), 'Values':val}
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                                       
                return [returnDateAll, returnValAll]

        
        def allRSI(self, period = 9, obLevel = 80, osLevel = 20):
                ts = self.rRsi(period)
                #print ts.Values(), 'RSI'

                returnValAll = []
                returnDateAll = []
                L = len(ts.Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        returnDate = ''
                        returnVal = '' 
                        if (len(ts.Dates())>0):
                                returnDate = ConvertDate(ts.Dates()[-nbBizDaysAgo-1]).isoformat()
                                #print 'rsi',ts.Values()[-1]
                                if ts.Values()[-nbBizDaysAgo-1]>obLevel:
                                        returnVal = 'Overbought'
                                elif ts.Values()[-nbBizDaysAgo-1]<osLevel:
                                        returnVal = 'Oversold'
                                else:
                                        returnVal = 'Neutral'
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                                     
                return [returnDateAll, returnValAll]

        #-2 for bearish; +2 for bullish -1 for oversold, +1 for overbought, 0 for neutral      
        
        def allStochastics(self, period = 10, obLevel = 80, osLevel = 20):
                tsD = self.rStoD(period)
                tsK = self.rStoK(period)
                #print tsD.Values(), '%D'
                #print tsK.Values(), '%K'

                returnValAll = []
                returnDateAll = []
                L = len(tsD.Dates())
                for nbBizDaysAgo in range(0,L):
                        
                        returnDate = ''
                        returnVal = ''
                        if (len(tsD.Dates())>0):
                                returnDate = ConvertDate(tsD.Dates()[-nbBizDaysAgo-1]).isoformat()
                                if tsD.Values()[-nbBizDaysAgo-1]>obLevel:
                                        returnVal = 'Overbought'
                                elif tsD.Values()[-nbBizDaysAgo-1]<osLevel:
                                        returnVal = 'Oversold'
                                elif (tsD.Values()[-nbBizDaysAgo-1]<=obLevel) and (tsD.Values()[-nbBizDaysAgo-1]>=osLevel) and tsK.Values()[-nbBizDaysAgo-1]>tsD.Values()[-nbBizDaysAgo-1]:
                                        returnVal = 'Bullish'
                                elif (tsD.Values()[-nbBizDaysAgo-1]<=obLevel) and (tsD.Values()[-nbBizDaysAgo-1]>=osLevel) and tsK.Values()[-nbBizDaysAgo-1]<tsD.Values()[-nbBizDaysAgo-1]:
                                        returnVal = 'Bearish'
                                else:
                                        returnVal = 'Neutral'
                                returnValAll.append(returnVal)
                                returnDateAll.append(returnDate)
                return [returnDateAll, returnValAll]

        #Internal method used in conjunction with lastDoji,KRD,Engulf,Hammer

        def allfilterSignal(self, tsBulls, tsBears,lag):
                bullLength = len(tsBulls.Dates())
                bearLength = len(tsBears.Dates())
                
                date = ''
                val = ''
                active = ''
                bullOrBear = ''

                SpotsDates = list(self.securityC().Dates())
                L = len(SpotsDates)
                
                Spots = list(self.securityC().Values())
                Highs = list(self.securityH().Values())
                Lows = list(self.securityL().Values())

                bullsdatesfull = tsBulls.Dates()
                bearsdatesfull = tsBears.Dates()
                
                bullsvaluesfull = tsBulls.Values()
                bearsvaluesfull = tsBears.Values()

                filterSigAllDates = []
                filterSigAllVals = []
                filterSigAllActive = []

                blankArray = [""]*len(SpotsDates)
                blankArrayBullDates = [""]*len(SpotsDates)
                blankArrayBearDates = [""]*len(SpotsDates)
                blankArrayBull =  [""]*len(SpotsDates)
                blankArrayBear =  [""]*len(SpotsDates)
                
                for k in bullsdatesfull:
                        
                        blankArray[SpotsDates.index(k)] = 'Bullish'
                        blankArrayBull[SpotsDates.index(k)] = 'Bullish'
                        blankArrayBullDates[SpotsDates.index(k)] = k

                for k in bearsdatesfull:
                        blankArray[SpotsDates.index(k)] = 'Bearish'
                        blankArrayBear[SpotsDates.index(k)] = 'Bearish'
                        blankArrayBearDates[SpotsDates.index(k)] = k

                
                
                datesvaluesfull = TimeSeries(dates = SpotsDates, values = blankArray)
                
                #valsBulls = Ether.Creator('TimeSerie', Dates = blankArrayBullDates, Values = blankArrayBull)
                #valsBears =  Ether.Creator('TimeSerie', Dates = blankArrayBearDates, Values = blankArrayBear)
                
                blankArrayNew = [""]*len(SpotsDates)
                INDEX = [0]*len(SpotsDates)
                
                for i in range(lag,L):
                        
                        if blankArray[i-lag] != "":
                                for j in range(i-lag,i):
                                       
                                        blankArrayNew[j]=blankArray[i-lag]
                                        INDEX[j] = i-lag

                for i in range(L-lag+1,L):
                        if blankArray[i] != "":
                                for j in range(i,L):
                                       
                                        blankArrayNew[j]=blankArray[i]
                                        INDEX[j] = i
                                        
                for i in range(0,L):
                        
                        if i < L-lag:
                                indend = INDEX[i] + lag
                        else:
                                indend = L-1
                                
                        date = SpotsDates[i]
                        if blankArrayNew[i] != "":
                                if blankArrayNew[i] == 'Bullish':
                                        close,endPos = determineExit(Spots,Highs[INDEX[i]],Lows[INDEX[i]],'stoploss', INDEX[i],indend,'Buy',INDEX[i])
                                        if (endPos == 0):
                                                active = 'Yes'
                                        else:
                                                active = 'No'
                                        bullOrBear = blankArrayNew[i]
                                        d = ConvertDate(date).isoformat()
                                else:
                                        close,endPos = determineExit(Spots,Highs[INDEX[i]],Lows[INDEX[i]],'stoploss', INDEX[i],indend,'Sell',INDEX[i])
                                        if (endPos == 0):
                                                active = 'Yes'
                                        else:
                                                active = 'No'
                                        bullOrBear = blankArrayNew[i]
                                        d = ConvertDate(date).isoformat()
                                                
                                        
                        else:
                                d = ''
                                bullOrBear = ''
                                active = ''
                        
                
                        filterSigDate = d
                        filterSigVal = bullOrBear
                        filterSigActive = active

                        filterSigAllDates.append(filterSigDate)
                        filterSigAllVals.append(filterSigVal)
                        filterSigAllActive.append(filterSigActive)

                       
                return [filterSigAllDates,filterSigAllVals,filterSigAllActive]
       
        def allDoji(self, lag = 25):
                
                tsBulls= self.bullDoji(maxHist)
                tsBears = self.bearDoji(maxHist)

                
                filterSig = self.allfilterSignal(tsBulls, tsBears,lag)   
                return filterSig

        
        def allKRD(self, lag = 25):
                tsBulls= self.bullKRD(maxHist)
                tsBears = self.bearKRD(maxHist)

                
                filterSig = self.allfilterSignal(tsBulls, tsBears,lag)   
                return filterSig
        
        
        def allEngulf(self, lag = 25):
                tsBulls= self.bullEngulf(maxHist)
                tsBears = self.bearEngulf(maxHist)

                
                filterSig = self.allfilterSignal(tsBulls, tsBears,lag)   
                return filterSig
       
        def allHammer(self, lag = 25):
                tsBulls= self.hammer(maxHist)
                tsBears = self.shootingStar(maxHist)

                
                filterSig = self.allfilterSignal(tsBulls, tsBears,lag)   
                return filterSig
        
# Determine exit based on whether the high is taken out (in a sell signal) or a low taken out (in a buy signal)
def determineExit(spotcloses, highPoint, lowPoint, exitCriteria, indstart, indend, buyOrSell, modifiedStart):

    #highs = self.securityH()
    #lows = self.securityL()
    #spots = self.securityC()
    endPos = 0
    comparePoint = 0

    if (buyOrSell == 'Buy'):
        comparePoint = lowPoint#self.LowValues()[indstart]#lows.Values()[indstart]
    elif (buyOrSell == 'Sell'):
        comparePoint = highPoint#self.HighValues()[indstart]#highs.Values()[indstart]
        
    if (exitCriteria != ''):
            for close in (spotcloses[modifiedStart+1:indend+1]): 
                    endPos = endPos+1
                    if ((buyOrSell == 'Sell') & (close > comparePoint)) or ((buyOrSell == 'Buy') & (close < comparePoint)):
                        return [close, endPos]
    return [spotcloses[indend], 0] #endPos is zero if not stopped out


# Determine entry based on a 33% pullback criteria. Returns the entry price and starting index of Entry
def determineEntry(spotcloses, highPoint, lowPoint, entryCriteria, indstart, indend, buyOrSell):

        
        #highs = self.securityH()
        #lows = self.securityL()
        #spots = self.securityC()
        startPos = 0
    
        startVal = spotcloses[indstart]
        #spots.Values()[indstart]
        #highPoint = self.HighValues()[indstart]#highs.Values()[indstart]
        #lowPoint = self.LowValues()[indstart]#lows.Values()[indstart]

        if (entryCriteria != 0):
                if (buyOrSell == 'Sell'):
                        entryPoint = startVal+(float(entryCriteria)*(highPoint-startVal)) 
                elif (buyOrSell == 'Buy'):
                        entryPoint = startVal-(float(entryCriteria)*(startVal-lowPoint))

                for close in (spotcloses[indstart+1:indend]):
                        startPos = startPos + 1
                        if ((buyOrSell == 'Sell') & (close > entryPoint) & (close <= highPoint)) or ((buyOrSell == 'Buy') & (close < entryPoint) & (close >= lowPoint)):
                                return [close, startPos]
                return [0, startPos] #No entry at all as per criteria 
        return [startVal, 0] #startPos is 0 if entry is on indstart ie no modified entry


def main():    
        print 'start', datetime.datetime.today()
        t=acTechSeries('USD','JPY',maxHist, 'D','True')
        #print t.rBolliupper(t.securityC())
        #u=fxtechseries('EUR','AUD',maxHist,'D')
        #print t.currentRSI(), 'original', t.securityC().Dates()[-1]
        #ts = t.getStartPart(2)
        #print ts.currentRSI(), 'startpart', ts.securityC().Dates()[-1]
        #print t.currentMACD()
        #print t.currentStochastics()
        #print t.allMean(21)
        print t.allADX()
        '''print t.allDoji(25)[1]
        print len(t.allDoji(25)[1])
        print len(t.securityC().Dates())'''
        #print t.allKRD()
        #print t.allEngulf()
        #print t.allHammer()
        #print t.lastKRD(36,30)
        #print t.lastDoji(36,30)
        #print t.lastEngulf(36,30)
        #print t.lastHammer(36,30)
        #print u.entrySignals('Buy', 'rRsi', 9, 30, True).Dates()
        #u=fxtechseries('EUR','CHF',maxHist,'W')
        #print '5000'
        #print u.lastEngulf(50)
        #print t.securityO.Values(), t.securityO.Dates()
        #print t.securityH.Values(), t.securityH.Dates()
        #print t.securityL.Values(), t.securityL.Dates()
        #print t.securityC.Values(), t.securityC.Dates()
        #print len(t.securityO.Values()), len(t.securityH.Values()), len(t.securityL.Values()), len(t.securityC.Values()), len(t.securityC.Dates())
        """print t.rStoD().Values()
        print t.currentMean(21)
        print t.maCrossOver(5,21)
        t.currentADX(20)
        print t.currentMACD()
        print t.currentRSI()
        print t.currentStochastics()
        print t.lastHammer()"""
        #print t.currentMean(6)
        #print t.lastEngulf(25)
        #print t.lastKRD()
        #t.currentMean(21)
        #print t.rMean(t.securityC,21).Values()
        #print t.rMaPositioning(5,21).Values()
        print 'done', datetime.datetime.today()



if __name__=="__main__":
	main()

                   
