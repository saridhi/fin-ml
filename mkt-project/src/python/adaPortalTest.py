import sys
import datetime, time
from techseriesNEW import maxHist
from techseriesNEW import acTechSeries
from heatmap import heatMapRank, toprank, bottomrank, getColorDico, selection_sort, absoluteRank
import logging
import time

#Load up the maximum possible history for efficient caching
class Startup():

    securities = {}
    securities['AUDCAD'] = maxHist
    securities['AUDJPY'] = maxHist
    '''securities['AUDNZD'] = maxHist
    securities['AUDUSD'] = maxHist
    securities['CADJPY'] = maxHist
    securities['CHFJPY'] = maxHist
    securities['EURAUD'] = maxHist
    securities['EURCAD'] = maxHist
    securities['EURCHF'] = maxHist
    securities['EURCZK'] = maxHist
    securities['EURGBP'] = maxHist
    securities['EURHUF'] = maxHist
    securities['EURJPY'] = maxHist
    securities['EURNOK'] = maxHist
    securities['EURNZD'] = maxHist
    securities['EURPLN'] = maxHist
    securities['EURSEK'] = maxHist
    securities['EURUSD'] = maxHist
    securities['GBPCAD'] = maxHist
    securities['GBPCHF'] = maxHist
    securities['GBPJPY'] = maxHist
    securities['GBPUSD'] = maxHist
    securities['NZDCAD'] = maxHist
    securities['NZDJPY'] = maxHist
    securities['NZDUSD'] = maxHist
    securities['USDCAD'] = maxHist
    securities['USDCHF'] = maxHist
    securities['USDCNY'] = maxHist
    securities['USDCZK'] = maxHist
    securities['USDHKD'] = maxHist
    securities['USDHUF'] = maxHist
    securities['USDINR'] = maxHist
    securities['USDJPY'] = maxHist
    securities['USDMXN'] = maxHist
    securities['USDNOK'] = maxHist
    securities['USDPLN'] = maxHist
    securities['USDSEK'] = maxHist
    securities['USDSGD'] = maxHist
    securities['USDTHB'] = maxHist
    securities['USDTRY'] = maxHist
    securities['USDZAR'] = maxHist
    # securities['XAGUSD'] = maxHist -- silver has a bug.
    securities['XAUUSD'] = maxHist'''
  
    def initSecurities(self):
        import cPickle as pickle
        total_tlength = 0.
        for s in self.securities.keys():
            #Init Daily
            t = acTechSeries(s[0:3],s[-3:], self.securities[s], 'D',False)
            fO = open(s[0:3]+s[-3:]+"O.txt",'w')
            fH = open(s[0:3]+s[-3:]+"H.txt",'w')
            fL = open(s[0:3]+s[-3:]+"L.txt",'w')
            fC = open(s[0:3]+s[-3:]+"C.txt",'w')
            fD = open(s[0:3]+s[-3:]+"D.txt",'w')
            t_start = time.time()
            t.RawData()
            t.isUptrend()
            t.isDowntrend()
            #t.rRegSlope(t.securityC())
            t_end = time.time()
            tlength = t_end-t_start
            total_tlength += tlength
            ptime = str(tlength).split('.')[0]+'.'+str(tlength).split('.')[1][0:2]
            #logx.Write('did ; '+s+', '+str(maxHist)+', in '+ptime+' seconds')
            ptotaltime = str(total_tlength).split('.')[0]+'.'+str(total_tlength).split('.')[1][0:2]
            pickle.dump(t.securityO().Values(),fO)
            pickle.dump(t.securityH().Values(), fH)
            pickle.dump(t.securityL().Values(), fL)
            pickle.dump(t.securityC().Values(), fC)
            pickle.dump(t.securityC().Dates(), fD)
            
        #logx.Write('Did the full list in ; '+ptotaltime+' seconds')
        #Libors = liborTS('USD','3m',maxHist)

    def dailyChanges(self, entries = 4):
        changesArray = []
        for s in self.securities.keys():
            t = fxtechseries(s[0:3],s[-3:], self.securities[s], 'D')
            changesArray.append((s,(t.securityC().Values()[-1]-t.securityC().Values()[-2])/t.securityC().Values()[-2]))

        sortedList = selection_sort(changesArray)
        topentries = []
        bottomentries = []          
        return toprank(sortedList),bottomrank(sortedList)
            
    def initPortalHeatMap(self):
        heatArray = []
        from pyroTechLib import pyroTechSeries
        pt  = pyroTechSeries()
        for s in self.securities.keys():
            heatDico = {}
            t = acTechSeries(s[0:3],s[-3:], self.securities[s], 'D')
            heatDico['securityName'] = s
            heatDico['MA1'] = pt.snapMA(asset = s[0:3], base = s[-3:], lag = 21)['Values']
            heatDico['MA2'] = pt.snapMA(asset = s[0:3], base = s[-3:], lag = 5)['Values']
            heatDico['MACrossover'] = pt.snapCrossOver(asset = s[0:3], base = s[-3:])['Values']
            heatDico['ADX'] = pt.snapADX(asset = s[0:3], base = s[-3:], lag = 7)['Values']
            heatDico['RSI'] = pt.snapRSI(asset = s[0:3], base = s[-3:], lag = 9)['Values']
            heatDico['Stochastics'] = pt.snapStochastics(asset = s[0:3], base = s[-3:])['Values']
            heatDico['MACD'] = pt.snapMACD(asset = s[0:3], base = s[-3:])['Values']
            heatDico['Doji'] = pt.snapDoji(asset = s[0:3], base = s[-3:])['Values']
            heatDico['Engulfing'] = pt.snapEngulfing(asset = s[0:3], base = s[-3:])['Values']
            heatDico['KRD'] = pt.snapKRD(asset = s[0:3], base = s[-3:])['Values']
            heatDico['Hammer/SS'] = pt.snapHammer(asset = s[0:3], base = s[-3:])['Values']
            heatDico['DojiActive'] = pt.snapDoji(asset = s[0:3], base = s[-3:])['Active']
            heatDico['EngulfingActive'] = pt.snapEngulfing(asset = s[0:3], base = s[-3:])['Active']
            heatDico['KRDActive'] = pt.snapKRD(asset = s[0:3], base = s[-3:])['Active']
            heatDico['Hammer/SSActive'] = pt.snapHammer(asset = s[0:3], base = s[-3:])['Active']
            heatArray.append(heatDico)
        return heatArray

    def getIndicatorValues(self,lag=25):

        secarray = {}
        print 'start loop' 
        for sec in self.securities.keys():
            indarray = {}
            
            t = acTechSeries(sec[0:3],sec[-3:], self.securities[sec], 'D', 'Pickle')
            
            
            indarray['MA1'] = t.allMean(21)
            print 'ma1'
            indarray['MA2'] = t.allMean(5)
            print 'ma2'
            indarray['MACrossover'] = t.allmaCrossOver(5,21)
            print 'macrossover'
            indarray['ADX'] = t.allADX(7,25)
            print 'adx'
            indarray['RSI'] = t.allRSI(9,80,20)
            print 'RSI'
            indarray['Stochastics'] = t.allStochastics()
            print 'Stochastics'
            indarray['MACD'] = t.allMACD()
            print 'MACD'
            indarray['Doji'] = t.allDoji(lag)
            print 'Doji'
            indarray['Engulfing'] = t.allEngulf(lag)
            print 'Engulfing'
            indarray['KRD'] = t.allKRD(lag)
            print 'KRD'
            indarray['Hammer/SS'] = t.allHammer(lag)
            print 'Hammer/SS'

            print 'start secarray'
            secarray[sec]=indarray
            print 'end secarray'
        return secarray

    def getIndicatorDate(self,ind,secarrayvalues,closeIndex):

        secarrayval = secarrayvalues[ind]
        secarrayval = secarrayval[1]
        
        secarrayvalCurrent = secarrayval[closeIndex]
        
        return secarrayvalCurrent

    def getIndicatorActiveDate(self,ind,secarrayvalues,closeIndex):

        secarrayval = secarrayvalues[ind]
        secarrayval = secarrayval[2]
        
        secarrayvalCurrent = secarrayval[closeIndex]
        
        return secarrayvalCurrent
    
    def backtestPortalHeatMap(self,secarray,start,end,period):
        heatMapHistoryT = {}
        heatMapHistoryB = {}
        
        #from pyroTechLib import pyroTechSeries
        #pt  = pyroTechSeries()
        for daysAgo in range(start,end):
            heatArray = []
            
            print "Testing for daysAgo:", daysAgo
            currentDate = datetime.datetime.today()-datetime.timedelta(daysAgo+1)
            for s in self.securities.keys():

                
                secarrayvalues = secarray[s]
                t = acTechSeries(s[0:3],s[-3:], self.securities[s], period, 'Pickle')
                
                heatDico = {}
                closeIndex = t.getOHLCIndices(ConvertDate(currentDate))[0]
                
                bizDaysAgo = len(t.securityC().Dates())-closeIndex-1
                
                print "Loaded security:", s
                
                
                if (closeIndex!=-1):
                    heatDico['securityName'] = s
                    
                    heatDico['MA1'] = self.getIndicatorDate('MA1',secarrayvalues,bizDaysAgo)
                    print "Testing for MA1:", heatDico['MA1']

                    heatDico['MA2'] = self.getIndicatorDate('MA2',secarrayvalues,bizDaysAgo)
                    print "Testing for MA2:", heatDico['MA2']

                    heatDico['MACrossover'] = self.getIndicatorDate('MACrossover',secarrayvalues,bizDaysAgo)
                    print "Testing for MACross:", heatDico['MACrossover']

                    heatDico['ADX'] = self.getIndicatorDate('ADX',secarrayvalues,bizDaysAgo)
                    print "Testing for ADX:", heatDico['ADX']

                    heatDico['RSI'] = self.getIndicatorDate('RSI',secarrayvalues,bizDaysAgo)
                    print "Testing for RSI:", heatDico['RSI']

                    heatDico['Stochastics'] = self.getIndicatorDate('Stochastics',secarrayvalues,bizDaysAgo)
                    print "Testing for Stochastics:", heatDico['Stochastics']

                    heatDico['MACD'] = self.getIndicatorDate('MACD',secarrayvalues,bizDaysAgo)
                    print "Testing for MACD:", heatDico['MACD']

                    heatDico['Doji'] = self.getIndicatorDate('Doji',secarrayvalues,closeIndex )
                    print "Testing for Doji:", heatDico['Doji']

                    heatDico['Engulfing'] = self.getIndicatorDate('Engulfing',secarrayvalues,closeIndex )
                    print "Testing for Engulfing:", heatDico['Engulfing']

                    heatDico['KRD'] = self.getIndicatorDate('KRD',secarrayvalues,closeIndex )
                    print "Testing for KRD:", heatDico['KRD']
                    
                    heatDico['Hammer/SS'] = self.getIndicatorDate('Hammer/SS',secarrayvalues,closeIndex )
                    print "Testing for Hammer:", heatDico['Hammer/SS']

                    heatDico['DojiActive'] = self.getIndicatorActiveDate('Doji',secarrayvalues,closeIndex)
                    print "Testing for DojiActive:", heatDico['DojiActive']

                    heatDico['EngulfingActive'] = self.getIndicatorActiveDate('Engulfing',secarrayvalues,closeIndex ) 
                    print "Testing for EngulfingActive:", heatDico['EngulfingActive']

                    heatDico['KRDActive'] = self.getIndicatorActiveDate('KRD',secarrayvalues,closeIndex )
                    print "Testing for KRDActive:", heatDico['KRDActive']

                    heatDico['Hammer/SSActive'] = self.getIndicatorActiveDate('Hammer/SS',secarrayvalues,closeIndex)
                    print "Testing for Hammer:", heatDico['Hammer/SS']
                    heatArray.append(heatDico)
            if len(heatArray) == 0:
                continue;
            valueList = heatValues(heatArray)
            
            sortedList = selection_sort(valueList)
            
            #topentries selection
            L = len(sortedList)
            markertop = 0
            for i in range(0,L-1):
                if sortedList[L-1][1] > 0:
                
                    if sortedList[L-1][1] == sortedList[L-2][1]:
                        if sortedList[L-1-i][1] == sortedList[L-2-i][1]:
                        
                            markertop = markertop + 1
                        else:
                            break
            entriestop = markertop + 1

            #bottomentries selection
           
            markerbottom = 0
            for i in range(0,L-1):
                if sortedList[0][1] < 0:
                
                    if sortedList[0][1] == sortedList[1][1]:
                        if sortedList[i][1] == sortedList[i+1][1]:
                        
                            markerbottom = markerbottom + 1
                        else:
                            break
            entriesbottom = markerbottom + 1
            
            
            topentries,bottomentries = toprank(sortedList,entriestop),bottomrank(sortedList,entriesbottom)
            print "topentries", topentries
            print "bottomentries", bottomentries
            
            for i in range(0,entriestop):
                s = topentries[i][0]
                heatMapHistoryT[(t.securityC().Dates()[closeIndex],i)]=s
               
            for i in range(0,entriesbottom):
                s = bottomentries[i][0]
                heatMapHistoryB[(t.securityC().Dates()[closeIndex],i)]=s
        
        return heatMapHistoryT,heatMapHistoryB

    def backtestHeatMap(self):
        heatArray = []
        for s in self.securities.keys():
            allHistory = []
            t = acTechSeries(s[0:3], s[-3:], 40)
            logging.info("Processing "+str(s)+"...")
            print 'Processing...', len(t.securityC().Dates())
            allMean1 = t.allMean(lag=21)
            allMean2 = t.allMean(lag=5)
            allMaCrosses = t.allMaCrossOver(5,21)
            allADX = t.allADX(lag=7)
            allRSI = t.allRSI(period=9)
            allStochastics = t.allStochastics(period=10)
            allMACD = t.allMACD()
            allDoji = t.allDoji(lag = 25)
            allEngulf = t.allEngulf(lag = 25)
            allKRD = t.allKRD(lag = 25)
            allHammer = t.allHammer(lag = 25)
            #minHist = min(len(allMean1), len(allMean2), len(allMaCrosses), len(allADX), len(allRSI), len(allStochastics), len(allMACD), len(allEngulf), len(allDoji), len(allKRD), len(allHammer))
            allHistory.append(allMean1)
            allHistory.append(allMean2)
            allHistory.append(allMaCrosses)
            allHistory.append(allADX)
            allHistory.append(allRSI)
            allHistory.append(allStochastics)
            allHistory.append(allMACD)
            allHistory.append(allDoji)
            allHistory.append(allEngulf)
            allHistory.append(allKRD)
            allHistory.append(allHammer)
            heatArray.append(allHistory)

        dateArray = []
        dates = []
        for dt in t.securityC().Dates():
            secArray = []
            datesNotThere = False
            count = 0
            for secs in heatArray:
                finalVerdict = []
                for indications in secs:
                    try:
                        finalVerdict.append([indications[str(dt)]])
                    except:
                        datesNotThere = True
                        print 'No data for date', dt, 'on indication', count
                        break
                if datesNotThere:
                    break
                newArray = []
                tempDict = {}
                for i in range(0,11):
                    if i==0:
                        tempDict['MA1']=finalVerdict[i][0][1]
                    elif i==1:
                        tempDict['MA2']=finalVerdict[i][0][1]
                    elif i==2:
                        tempDict['MACrossover']=finalVerdict[i][0][1]
                    elif i==3:
                        tempDict['ADX']=finalVerdict[i][0][1]
                    elif i==4:
                        tempDict['RSI']=finalVerdict[i][0][1]
                    elif i==5:
                        tempDict['Stochastics']=finalVerdict[i][0][1]
                    elif i==6:
                        tempDict['MACD']=finalVerdict[i][0][1]
                    elif i==7:
                        tempDict['Doji']=finalVerdict[i][0][1]
                        tempDict['DojiActive']=finalVerdict[i][0][2]
                    elif i==8:
                        tempDict['Engulfing']=finalVerdict[i][0][1]
                        tempDict['EngulfingActive']=finalVerdict[i][0][2]
                    elif i==9:
                        tempDict['KRD']=finalVerdict[i][0][1]
                        tempDict['KRDActive']=finalVerdict[i][0][2]
                    elif i==10:
                        tempDict['Hammer/SS']=finalVerdict[i][0][1]
                        tempDict['Hammer/SSActive']=finalVerdict[i][0][2]
                tempDict['securityName']=self.securities.keys()[count]
                count = count + 1
                secArray.append(tempDict)
            if datesNotThere:
                continue
            if (secArray!=[]):
                dates.append(dt)
                #print len(secArray)
                #print len(secArray[0]), len(secArray[1])
                dateArray.append(secArray)
        #print dateArray
        return dateArray, dates

    def backtestTopEntries(self,heatMapHistory,L=25):
        
        t = Backtest()
        dicobacktest= {}
        dicobacktest['Length']=L
        dates = []
        sec = heatMapHistory.values()

        for dateskeys in heatMapHistory.keys():
            date = dateskeys[0]
            dates.append(date)
        
        for i in range(1,L+1):
            securities,dicodateret = t.createReturnsPairs(heatMapHistory,i)
            returns = []
            for j in range(0,len(securities)):
                
                dicosecdateret=dicodateret[j]
                returns.append(dicosecdateret.values())
            dicobacktest[i]=returns # for each i , print returns over next i days corresponding to each security in sec and date in dates  
        
        
        return sec,dates,dicobacktest
         
    def createTable(self, topentries, bottomentries, topgain, toploss):
        rowcolor = []
        bgcolors = []
        table = []
        row = []
        row.append('Bullish')
        rowcolor.append('')
        
        for a,b in topentries:
            row.append(a)
            rowcolor.append(getColorDico()[str(b)])
        bgcolors.append(rowcolor)
        table.append(row)
        row = []
        rowcolor = []
        row.append('Bearish')
        
        rowcolor.append('')
        for a,b in bottomentries:
            row.append(a)
            rowcolor.append(getColorDico()[str(b)])
            #rowcolor.append('')
        bgcolors.append(rowcolor)
        table.append(row)
        row = []
        rowcolor = []
        row.append('Max daily gain(%)')
        rowcolor.append('')
        
        for a,b in topgain:
            row.append(a)
            rowcolor.append('')
        bgcolors.append(rowcolor)
        table.append(row)
        row = []
        rowcolor = []
        row.append('Max daily loss(%)')
        rowcolor.append('')
        for a,b in toploss:
            row.append(a)
            rowcolor.append('')
        bgcolors.append(rowcolor)
        table.append(row)

        return table, bgcolors

    def getHelpText(self):
        xmlstring = ''
        xmlstring+= '<p>This heatmap gives forecasting indications for assets<p>'
        return xmlstring

def adaPortal():
        adaPortal = Startup()
        dateArray, dates = adaPortal.backtestHeatMap()
        f = open('backtest.csv', 'w')
        
        for secArray, dt in zip(dateArray, dates):
            barMenu, allDrinks = absoluteRank(secArray)
            print barMenu, dt
            f.write(str(barMenu)+','+str(dt)+'\n')

        f.close()
        print 'done', datetime.datetime.today()
        
if __name__=="__main__":
	adaPortal()












