import sys
#import Pyro.core
#import Pyro.naming
#from Pyro.errors import NamingError
#import pythoncom
import threading
import datetime
from threading import currentThread
from techseriesNEW import maxHist, firstLegitDate, acTechSeries as fxtechseries
import techbacktestNEW
import techlib
#import distribution
import ConfigParser, os
import startupNEW
import heatmap
#import heatmap2
import logging
#from graphelements import ConvertDate
LOG_FILENAME = 'tech.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def todico(L):
        try:
                return {'Dates':L[0], 'Values':L[1]}
        except:
                return L

def todico2(L):
        try:
                return {'Dates':L[0], 'Values':L[1], 'Active':L[2]}
        except:
                return L

#Snap functions return the last value of the timeseries as a hash table of date and value
class pyroTechSeries():
        
        # Constructor
        '''def __init__(self):
                Pyro.core.ObjBase.__init__(self)
	'''
        def initCache(self):
                #pythoncom.CoInitialize()
                cacheInit=startup.Startup()
                cacheInit.initSecurities()        

        def heatMap(self, securityList = []):
                #pythoncom.CoInitialize()
                return heatmap.heatMap(securityList)

        #Returns the date of the last Hammer/ShootingStar and whether it was a bullish or bearish signal
        def heatMap2(self, securityList = [{'securityName':'EUR/USD', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'Yes','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''}]):
                #pythoncom.CoInitialize()
                return heatmap2.heatMap(securityList)

        #Returns the sequence of all possible colors in an array- the hex code in order of bearish to bullish
        def heatLegend(self):
                #pythoncom.CoInitialize()
                return heatmap.getColorArray()

        # Returns values are 1 if bullish -1 if bearish
        def snapMA(self, asset='AUD', base='JPY', lag=21, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = lag+100
                else:
                        horizon= (lag*5)+100'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico(t.currentMean(lag))

        #To determine MA crossover 1 returned for bullish -1 for bearish
        def snapCrossOver(self, asset='EUR', base='CHF', fastMaVal = 5, slowMaVal = 21, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = slowMaVal+100
                else:
                        horizon= (slowMaVal*5)+100'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico(t.maCrossOver(fastMaVal, slowMaVal))

        #ADX 1 returned if above the level specified else 0
        def snapADX(self, asset='EUR', base='USD', lag = 7, level = 25, period='D'):
                #pythoncom.CoInitialize()
                if(period=='D'):
                        horizon = 600
                else:
                        horizon= 3000
                t = fxtechseries(asset, base, horizon, period)
                return todico(t.currentADX(lag, level))

        #RSI 1 returned if above the obLevel specified else -1 if below osLevel else 0
        def snapRSI(self, asset='USD', base='JPY', lag = 7, obLevel = 75, osLevel = 25, period='D'):
                #pythoncom.CoInitialize()
                if(period=='D'):
                        horizon = 600
                else:
                        horizon= 3000
                t = fxtechseries(asset, base, horizon, period)
                return todico(t.currentRSI(lag, obLevel, osLevel))

        #MACD 1 returned if crossover above the 0 level, -1 if crossover below the 0 else 0
        def snapMACD(self, asset='USD', base='JPY', fastMaVal = 13, slowMaVal = 26, period='D'):
                #pythoncom.CoInitialize()
                if(period=='D'):
                        horizon = 600
                else:
                        horizon= 3000
                t = fxtechseries(asset, base, horizon, period)
                return todico(t.currentMACD(fastMaVal, slowMaVal))

        #Stochastics 1 returned if above the obLevel specified else -1 if below osLevel else 0
        def snapStochastics(self, asset='EUR', base='AUD', lag = 10, obLevel = 80, osLevel = 20, period='D'):
                #pythoncom.CoInitialize()
                if(period=='D'):
                        horizon = 600
                else:
                        horizon= 3000
                t = fxtechseries(asset, base, horizon, period)
                return todico(t.currentStochastics(lag, 0.80, 0.20))

        #Returns the date of the last Doji and whether it was a bullish or bearish signal
        def snapDoji(self, asset='USD', base='JPY', lag = 25, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = lag+250
                else:
                        horizon= (lag*7)+250'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico2(t.lastDoji(lag))
                
        #Returns the date of the last Key Reversal Day and whether it was a bullish or bearish signal
        def snapKRD(self, asset='USD', base='JPY', lag = 25, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = lag+250
                else:
                        horizon= (lag*7)+250'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico2(t.lastKRD(lag))

        #Returns the date of the last Engulfing and whether it was a bullish or bearish signal
        def snapEngulfing(self, asset='USD', base='JPY', lag = 25, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = lag+250
                else:
                        horizon= (lag*7)+250'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico2(t.lastEngulf(lag))

        #Returns the date of the last Hammer/ShootingStar and whether it was a bullish or bearish signal
        def snapHammer(self, asset='AUD', base='USD', lag = 25, period='D'):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = lag+250
                else:
                        horizon= (lag*7)+250'''
                t = fxtechseries(asset, base, maxHist, period)
                return todico2(t.lastHammer(lag))

        #Returns the Success rate of the signal amongst other stats
        def snapQuickStats(self, asset='GBP', base='USD', typeT = 'Bearish', signal = 'Doji', period = 'D',
                            tenorDate = datetime.date(2008,11,17)):
                #pythoncom.CoInitialize()
                t = fxtechseries(asset, base, maxHist, period)
                tenor = techbacktest.getTenorFromDate(t, tenorDate)
                startInd = [ConvertDate(d) for d in t.securityC().Dates()].index(ConvertDate(tenorDate)) 
                startVal = t.securityC().Values()[startInd]
                extremeReturn = 0
                if typeT == 'Bullish':
                        typeT = 'Buy'
                        maxVal = max(t.securityC().Values()[startInd:])
                        extremeReturn = (maxVal - startVal)/startVal
                else:
                        typeT = 'Sell'
                        minVal = min(t.securityC().Values()[startInd:])
                        extremeReturn = (startVal-minVal)/startVal

                if (extremeReturn <0):
                        extremeReturn= 0
                
                bt = techbacktest.hPerformance(t, [{'SignalName':signal, 'Value1':5, 'Value2':21, 'Type':typeT, 'Operator':'OR'}],tenor, maxHist, None,0,True)
                bt['ExtremeVal']=extremeReturn
                bt['CTMoves']=bt['CTMoves'][0]
                bt['Success']=bt['Success'][0]
                def convertToArray(bt):
                        returnArray = []
                        returnArray.append(bt)
                        return returnArray
                
                return convertToArray(bt)
        

        #Backtest per security but can handle multiple signals
        #Trigger is the trigger for trade entry  - it refers to a method in TechLib and should be in the following format:
        # ['bull'|'bear'].<method name from TechLib>
        # Horizon is number of days prior to start backtest from
        def backtest(self, asset='USD', base='JPY', paramsArray=[], horizon=360, period='D', tenor = 10, calculateAvg = False, pullback = 0.0, stoploss = True):
                #pythoncom.CoInitialize()
                '''if(period=='D'):
                        horizon = horizon+200
                else:
                        horizon= horizon+1000'''
                t = fxtechseries(asset, base, maxHist, period)
                return techbacktest.hPerformance(t, paramsArray, tenor, horizon, calculateAvg, pullback, stoploss)

        #Returns backtest results for a portfolio -- horizon is the startDate from which backtest is to return results
        #Tenor is xd or xdays depending on a return of a series of average daily returns or Avg daily returns on a single day
        def backtestPortfolio(self, assetArray=[], paramsArray=[{'SignalName':'Engulfing', 'Value1':21, 'Value2':5, 'Type':'Bearish', 'Operator':'OR'},
                                                                {'SignalName':'Stochs cross', 'Value1':10, 'Value2':5, 'Type':'Bullish', 'Operator':'AND'}],
                              startDate=datetime.date(1995,12,11), period='D', tenor = 25, calculateAvg = False, pullback = 0.0, stoploss = False):
                
                #pythoncom.CoInitialize()

                logging.info(paramsArray)
                logging.info(assetArray)
                logging.info(startDate)
                logging.info(period)
                logging.info(tenor)
                logging.info(calculateAvg)

                from techseries import today
                horizon = (today - startDate).days
                
                assets = []
                bases = []

                for secs in assetArray:
                        fx1 = secs.split('/')
                        assets.append(fx1[0])
                        bases.append(fx1[1])
                        
                returnArray = []
                dicoResult = {}
                allDates = []
                allValues = []
                allSuccess = []
                allReversal = []
                counter = 0

                for a,b in zip(assets, bases): #For each security
                        returnArray.append(self.backtest(a,b, paramsArray, horizon,period,tenor,calculateAvg,pullback,stoploss))
                
                #Calculate the average daily returns rather than spitting out all the signal data points
                if (calculateAvg) or (calculateAvg == None):
                        for n in (returnArray[0]['Days']): #Go through different tenors
                                allDates.append(n)
                                sizes = 0
                                totals = 0
                                successes = 0
                                reversals = 0
                                
                                for avgDict in returnArray: #Go through each currency pair in portfolio
                                        totals = (float(avgDict['Returns'][counter])*float(avgDict['Length'][counter]))+totals
                                        sizes = sizes + float(avgDict['Length'][counter])
                                        successes = (float(avgDict['Success'][counter])*float(avgDict['Length'][counter]))+successes
                                        reversals = (float(avgDict['Reversal'][counter])*float(avgDict['Length'][counter]))+reversals
                                counter = counter+1 #counter for number of days
                                
                                if (sizes!=0):
                                        allValues.append(totals/sizes)
                                        allReversal.append(reversals/sizes)
                                        allSuccess.append(successes/sizes)
                                else:
                                        allValues.append(0)
                                        allReversal.append(0)
                                        allSuccess.append(0)
                                

                        dicoResult = {'Days': allDates, 'Returns': allValues, 'Success':allSuccess, 'Reversal':allReversal}   
                else: #Spit out all the datapoints
                        for tsDict in returnArray:
                                allDates = tsDict['Dates']+allDates
                                allValues = tsDict['Returns']+allValues
                                allSuccess = tsDict['Success']+allSuccess
                                allReversal = tsDict['Reversal']+allReversal

                        #Add values for dates that are repeated
                        tempCommonDico = {}
                        exists = 0.0
                        for commonDt,commonVal in zip(allDates,allValues):
                                try:
                                        exists = tempCommonDico[commonDt]
                                        tempCommonDico[commonDt] = commonVal + exists
                                except:
                                        tempCommonDico[commonDt] = commonVal
                        allDates = tempCommonDico.keys()
                        allValues = tempCommonDico.values()

                        #Calculate Sharpe
                        sharpeRatio = techbacktest.calculateSharpe(allValues, allDates, tenor)
                        #sharpeRatio = 0 #For now default to 0
                        allDatesIso = [ConvertDate(dt).isoformat() for dt in allDates]
                        dicoResult = {'Dates':allDatesIso,'Returns':allValues,'Success':allSuccess,'Reversal':allReversal, 'Sharpe':sharpeRatio}

                return dicoResult

                
def main():
        t = pyroTechSeries()
        print 'start', datetime.datetime.today()
        #print t.snapKRD('EUR','JPY',25)
        #print t.snapStochastics('EUR','AUD',10,80,20,'D')
        #print t.backtestPortfolio(['AUD/USD', 'AUD/JPY'])
        xmlString = "<portfolio name='default' period = 'D'><Securities><Asset name='NZD/USD' dirty='true'/>"
        xmlString += "<Asset name='AUD/USD' dirty='false'/><Asset name='GBP/USD' dirty='false'/><Asset name='USD/CHF' dirty='false'/>"
        xmlString += "<Asset name='USD/JPY' dirty='false'/><Asset name='EUR/CHF' dirty='false'/><Asset name='EUR/GBP' dirty='false'/>"
        xmlString += "<Asset name='EUR/JPY' dirty='false'/><Asset name='EUR/USD' dirty='false'/><Asset name='USD/CAD' dirty='true'/>"
        xmlString += "</Securities><Indicators name='Indicators'><Indicator name='ADX' dirty = 'true'><Parameter name='TrendingLevel'>"
        xmlString += "<Value name='30' min='1' max='99'/></Parameter><Parameter name='Period'><Value name='7' min='1' max='40'/></Parameter>"
        xmlString += "</Indicator><Indicator name='RSI' dirty='true'><Parameter name='overbought'><Value name='75' min='51' max='99'/>"
        xmlString += "</Parameter><Parameter name='oversold'><Value name='25' min='1' max='40'/></Parameter><Parameter name='period'>"
        xmlString += "<Value name='9' min='1' max='100'/></Parameter></Indicator><Indicator name='MACrossover' dirty='true'><Parameter name='Fast'>"
        xmlString += "<Value name='5' min='0' max='99'/></Parameter><Parameter name='Slow'><Value name='21' min='1' max='40'/></Parameter></Indicator>"
        xmlString += "<Indicator name='Doji' dirty='true'><Parameter name='doji'><Value name='25' min='1' max='200'/></Parameter></Indicator>"
        xmlString += "<Indicator name='Engulfing' dirty='true'><Parameter name='engulfing'><Value name='25' min='1' max='200'/></Parameter>"
        xmlString += "</Indicator><Indicator name='KRD' dirty='true'><Parameter name='krd'><Value name='25' min='1' max='200'/></Parameter>"
        xmlString += "</Indicator></Indicators></portfolio>"
        print t.runSnapshot(xmlString)
        '''print t.snapDoji('USD','CHF',25,'D'
        print startup.Startup.maxHist
        print 'end first call of snapDoji', datetime.datetime.today()
        print t.snapDoji('USD','CHF',25,'D')
        print 'end second call of snapDoji', datetime.datetime.today()'''
        print 'done', datetime.datetime.today()

if __name__=="__main__":
	main()






