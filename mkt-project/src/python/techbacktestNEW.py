# -*- coding: cp1252 -*-

#from __future__ import with_statement

import sys


#from ether import Ether
#import ether
#from graphelements import ConvertDate
import datetime, time
##import pythoncom
import math
#import qadefs
#import days
#import threading
#import Queue
from techseriesNEW import determineExit, determineEntry, maxHist, acTechSeries
##import#win32com.client

userInputMap = {}

userInputMap['Doji in a downtrend']='bullDoji'
userInputMap['Doji']='Doji'
userInputMap['KRD']='KRD'
userInputMap['Engulfing']='Engulf'
userInputMap['Bullish Key Reversal']='bullKRD'# PK change: 'Key Reversal' is now 'KRD'
userInputMap['Hammer']='hammer'
userInputMap['Shooting Star']='shootingStar'
userInputMap['Bullish price/MA cross']='rMean'
userInputMap['RSI cross']='rRsi'
userInputMap['Overbought Bearish Stochastics Crossover']='rStoD'
userInputMap['Oversold Bullish Stochastics Crossover']='rStoD'
userInputMap['Bullish dual MA cross']='rMean'  #The difference in this is that both val1 and val2 are passed in
userInputMap['Bullish dual MA']='rMean'#The difference in this is that both val1 and val2 are passed in
userInputMap['Above']='Buy'
userInputMap['Below']='Sell'
userInputMap['Price above MA']='rMean'
userInputMap['Price below MA']='rMean'
userInputMap['ADX above level']='rADX'
userInputMap['ADX below level']='rADX'
userInputMap['Bullish']='Buy'
userInputMap['Bearish']='Sell'
userInputMap['Neutral']=''
userInputMap['Sell']='Sell'
userInputMap['Buy']='Buy'
userInputMap['RSI']='rRsi'
userInputMap['Overbought RSI']='rRsi'
userInputMap['Oversold RSI']='rRsi'
userInputMap['Stochs']='rStoD'
userInputMap['Bearish Stochastics']='rStoD'
userInputMap['Bullish Stochastics']='rStoD'
userInputMap['MA']='rMean'
userInputMap['Dual MA']='rMean'#The difference in this is that both val1 and val2 are passed in
userInputMap['MACrossover']='rMean'#The difference in this is that both val1 and val2 are passed in
userInputMap['MACD']='rVelocity'

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
#                                 Function for the Technical backtesting web page
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


#from techlib import QAAdddayexitToDate

def getdayexitFromDate(t, refdate=datetime.date.today()): 

    Dates = [ConvertDate(date) for date in t.securityC().Dates()]
    
    try:
        dayexit = len(Dates)-(Dates.index(refdate))
    except:
        dayexit = (datetime.date.today()-refdate).days
    return dayexit

#dayexit
def adddayexitToDate(Dates, refdate=datetime.date.today(), dayexit = 10):#To be implemented
    bizDaysToAdd = dayexit
    #Dates = [ConvertDate(date) for date in t.securityC().Dates()]
    
    indStart = Dates.index(ConvertDate(refdate))
    indEnd = indStart + int(dayexit)
    try:
        return Dates[indEnd]
    except:
        return refdate + datetime.timedelta(int(bizDaysToAdd))

def compareAndArrays(signalSet = [], andSignalSet = [], t = None):
    mdates = []
    newSignalInfo = []
    match = False
    dts = t.securityC().Dates()
    for s in signalSet:
        anddates = []
        andvalues = []
        mdates = []
        sigs = s[0]
        bOrb = s[1]
        op = s[2]
        vals = sigs.Values() #Entry criteria values

        #Does the (date-1 biz day) exist in the pre condition criteria array
        for v in vals:
            d = dts[v-1]
            for compareArray in andSignalSet:
                dtCompare = compareArray[0]
                if dtCompare.HasDate(d):
                    match = True
                    break
            if (match):
                mdates.append(dts[v])
                match = False

        #Find values for the legit dates
        for d in mdates:
            andval = sigs.Fetch(d)
            anddates.append(d)
            andvalues.append(andval)
            
        if (len(mdates) != 0):
            signals = Ether.Creator('TimeSerie', Dates = anddates, Values = andvalues)
            newSignalInfo.append((signals, bOrb, op))
    return newSignalInfo
            
    
#Filter common dates if andEntryOrPrecondition = AND
def splitAnds(signalInfo = [], t = None):
    preConditions = []
    entryConditions = []

    #Find first explicit signal without xOver = False
    for sInfo in signalInfo:
        
        if (sInfo[2]=='Precondition'):
            preConditions.append(sInfo)
        else:
            entryConditions.append(sInfo)

    newSignalInfo = compareAndArrays(entryConditions, preConditions,t)
    return newSignalInfo
    
#Performance backtest for multiple signals and this can also calculate Average Daily Returns
#For Reversal patterns pass in a value of 21 for values
#CalculateAvg = None for Snapshot page
def hPerformance(t, paramsArray = [{'SignalName':'Engulfing', 'Value1':9, 'Value2':75, 'Type':'Sell', 'Operator':'Entry Signal'}],
                 dayexit=10, horizon = maxHist, calculateAvg = False, pullback = 0.0, stoploss = True):

    signalInfo = []
    avgReturnArray = []
    msuccess = []
    mreversal = []
    mctmoves = []
    sizes = []
    dicoResult = {}
    crossOver = True
    andEntryOrPrecondition = 'Entry Signal'

    for params in paramsArray:
        signal = params['SignalName']
        value = params['Value1']
        level = params['Value2']
        inputTradeType = params['Type']
        #ttype = userInputMap[inputType]
        operator = params['Operator']
        
        if (operator == 'Precondition'): # Don't only include the specific date on which the crossover occurs
            crossOver = False #For pre-conditions the crossover should be false
            andEntryOrPrecondition = 'Precondition'
        else:
            crossOver = True
        method = getattr(t, 'entrySignals')
        signals = method(inputTradeType, userInputMap[signal], value, level, crossOver) #Should correspond with method name in TechLib
        
        signalInfo.append((signals, inputTradeType, operator)) #An array of lists with the signal information

        
    if (len(signalInfo)==1) or (andEntryOrPrecondition == '') or (andEntryOrPrecondition == 'Entry Signal'): #No AND or OR
        signals = signalInfo

           
    elif (andEntryOrPrecondition == 'Precondition'): #Using pre-conditions
        signals = splitAnds(signalInfo,t)

    
    if (len(signals) == 0):
        return {'Days': 0, 'Returns': [], 'Success':[], 'Reversal':[], 'Length':0, 'Dates':[]}
       
    if (calculateAvg):  #calculate Average can be for a single or multiple days
            rangeNumbers= range(1,dayexit+1)
            rangeOfDays= rangeNumbers
            
    else:
        rangeOfDays = [dayexit] #If not calculating average daily return then returns data points for a single day
            
    for dayLoop in rangeOfDays:
        dicoResult=calculateReturns(t, signals, dayLoop, horizon, pullback, stoploss)
        if (calculateAvg!=False):
            size = len(dicoResult['Returns'])
            sizes.append(size)
            if (size == 0):
                avgReturnArray.append(0)
                msuccess.append(0)
                mreversal.append(0)
            else:
                avgReturnArray.append((sum(dicoResult['Returns'])/size)/int(dayLoop))
                msuccess.append((float(len([z for z in dicoResult['Success'] if z==1]))/float(size)))
                mreversal.append((float((len([z for z in dicoResult['Reversal'] if z==1])))/float(size)))
                        
    if (calculateAvg == None) and len(dicoResult)!=0: #For snapshot page
        ctSize = len([z for z in dicoResult['CTMoves'] if z!=0])
        avgCtMove = 0
        if ctSize != 0:
            avgCtMove = (sum(dicoResult['CTMoves']))/ctSize
        mctmoves.append(avgCtMove)  
        dicoResult = {'Days': rangeOfDays, 'Returns': avgReturnArray, 'Success':msuccess, 'Reversal':mreversal, 'Length':sizes, 'CTMoves':mctmoves} 
        
    if (calculateAvg): #For backtest calculating average daily returns
        dicoResult = {'Days': rangeOfDays, 'Returns': avgReturnArray, 'Success':msuccess, 'Reversal':mreversal, 'Length':sizes}

    return dicoResult
    


#Success/Reversal rate assumes stoploss always, though the pullback matters for this calculation
# If dayexit ends with 'days' then get average daily return time series. Horizon is in calendar days, dayexit is in business days
def calculateReturns(t, signalInfo, dayexit=10, horizon = 360, entryCriteria = 0.0, stoploss = True):

    #print 'Calling calculateReturns with : ', t, si gnalInfo, dayexit, horizon
    
    #t.readFile() #this line is just for testing purposes
    avgReturns = {}
    startpoint = datetime.date.today() - datetime.timedelta(horizon)
    spots = t.securityC()
    tradeDates = []
    tradePayouts = []
    counterTrendMoves = []
    success = []
    reversal = []

    Dates = [ConvertDate(date) for date in spots.Dates()]
    Spots = list(spots.Values())
    Highs = list(t.securityH().Values())
    Lows = list(t.securityL().Values())

    for s in signalInfo:
        signals = s[0]
        buyOrSell = s[1]
        if (stoploss == True):
            exitCriteria = 'stoploss'
        else:
            exitCriteria = ''

        if ((signals != None) and (len(signals.Values())!= 0)):
            for indstart in (signals.Values()):

                datestart = Dates[indstart]
                dateend = adddayexitToDate(Dates, datestart, dayexit)

                if (dateend <= Dates[-1]) and (datestart >= startpoint):
                    spotstart = Spots[indstart]
                    if (spotstart <> 0):
                        try:
                            indend = Dates.index(dateend)
                        except:
                            break

                        firstPrice, startPos = determineEntry(Spots,Highs[indstart],Lows[indstart],entryCriteria, indstart, indend, buyOrSell)
                        
                        if ((firstPrice != 0) & (startPos != (indstart-indend))): #If the signal has kicked in - kicking in on the last day doesn't count
                            modifiedStart = indstart + startPos
                            lastPrice, endPos = determineExit(Spots,Highs[indstart],Lows[indstart],exitCriteria, indstart, indend, buyOrSell, modifiedStart)
                            
                            #Always use stoploss for success and reversal rate calcs below
                            lastPriceRates, endPosRates = determineExit(Spots,Highs[indstart],Lows[indstart],'stoploss', indstart, indend, buyOrSell, modifiedStart) 
                            if (endPosRates==0): #signal valid till last day -- this is used for success and reversal rate calcs
                                success.append(1)
                                lastIndex = indend + 1
                            else:
                                success.append(0)
                                lastIndex = endPosRates + modifiedStart + 1

                            #Cumulative Reversal Rate and Counter Trend moves
                            Values = Spots[modifiedStart+1:lastIndex]
                            reversalVal = 0
                            ctDistance = 0
                            if (buyOrSell == 'Buy'):
                                maxVal = max(Values)
                                if (maxVal>firstPrice): 
                                    reversalVal = 1
                                    ctDistance = (maxVal - firstPrice)/firstPrice
                            elif (buyOrSell == 'Sell'):
                                minVal = min(Values)
                                if (minVal<firstPrice):
                                    reversalVal = 1
                                    ctDistance = (firstPrice - minVal)/firstPrice
                            reversal.append(reversalVal)
                            counterTrendMoves.append(ctDistance)
                                
                            spotreturn = (lastPrice-firstPrice)/firstPrice
                            if (buyOrSell == 'Sell'):
                                spotreturn = -spotreturn
                            tradePayouts.append(spotreturn)
                            
                            '''if endPos == 0:
                                tradeDates.append(dateend)
                            else:
                                tradeDates.append(Dates[endPos + modifiedStart + 1])'''#Gets the close date rather than open
                            tradeDates.append(datestart)

    dicoResult = {'Dates':tradeDates,'Returns':tradePayouts, 'Success': success, 'Reversal': reversal, 'CTMoves':counterTrendMoves}
    return dicoResult

def calculateSharpe(arrayReturns = [], arrayDates = [], nbdays = 0):
    if len(arrayReturns) == 0:
        return 0.
    
    from numpy import std,sqrt
    #from fxstdtimeserie import liborTS
    from techseries import maxHist
    
    avgDailyReturn = (sum(arrayReturns)/len(arrayReturns))/int(nbdays)

    annReturn = avgDailyReturn * 250

    Libors = liborTS('USD','3m',maxHist)       
    totalLibor = 0

    for d in arrayDates:
        try:
            val = Libors.Fetch(d)
        except:
            val = Libors.Values()[-1] #If Libor doesn't exist default to today Libor
        totalLibor = val + totalLibor
    avgLibor = totalLibor/len(arrayDates)

    sd = std(arrayReturns)
    annSd = sqrt(250)*sd
    
    sharpe = (annReturn - (avgLibor/100))/annSd
    return sharpe

def main():
        from techseries import acTechSeries
        t = acTechSeries('EUR', 'GBP', 6156, 'D')
        #t.RawData()
        #calculateSharpe([2,3,4], [)
        #print 'got the ac data'
        #t.isDowntrend()
        #t.isUptrend()
        #print 'got the up and down trends'
        #print hPerformance(t,paramsArray = [{'SignalName':'Doji in a downtrend', 'Value1':25, 'Value2':0, 'Type':'Buy', 'Operator':'Entry Signal'}])
        #print hPerformance(t,paramsArray = [{'SignalName':'ADX above level', 'Value1':7, 'Value2':20, 'Type':'Buy', 'Operator':'Entry Signal'}])
        print hPerformance(t,paramsArray = [{'Operator': 'Entry Signal', 'Type': 'Buy', 'Value2': 0, 'Value1': 25, 'SignalName': 'RSI'}, {'Operator': 'Precondition', 'Type': 'Buy', 'Value2': 20, 'Value1': 7, 'SignalName': 'ADX above level'}],dayexit=25,horizon=4854,calculateAvg=True,pullback=0.0,stoploss=False)
        print hPerformance(t,paramsArray = [{'SignalName':'Doji in a downtrend', 'Value1':25, 'Value2':0, 'Type':'Buy', 'Operator':'Entry Signal'},
                                                                {'SignalName':'Bullish Key Reversal', 'Value1':7, 'Value2':20, 'Type':'Buy', 'Operator':'Precondition'}],dayexit=25,horizon=4854,calculateAvg=True,pullback=0.0,stoploss=False)

if __name__=="__main__":
	main()





        
