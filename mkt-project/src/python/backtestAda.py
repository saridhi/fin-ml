import shutil
import os
import commands
import smtplib
import email.utils
import getpass
import sys
import datetime
import time
import random
import logging

LOG_FILENAME = os.environ['LOG_DIR'] + '/historicada-%s.log' %str(datetime.date.today())
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

from randomSong import RandomSong
'''from SuperModelRecords import SuperModelRecords'''
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from techseriesNEW import maxHist, acTechSeries
from heatmap import absoluteRank, toprank, bottomrank, getColorDico, selection_sort        
'''from AdaSuperModel import AdaSMMain'''

def FillSMR(smr, h):
    attrib_str = ''
    for a in attribs:
        attrib_str = attrib_str + ',' + h[a]
    smr.SetAttribString(attrib_str)
    

class StartupHistoric():

    securities = {}
    securities['EURUSD'] = maxHist
    securities['AUDCAD'] = maxHist
    securities['AUDJPY'] = maxHist
    securities['AUDNZD'] = maxHist
    securities['AUDUSD'] = maxHist
    securities['CADJPY'] = maxHist
    securities['CHFJPY'] = maxHist
    securities['EURAUD'] = maxHist
    securities['EURCAD'] = maxHist
    securities['EURCHF'] = maxHist
    #securities['EURCZK'] = maxHist
    securities['EURGBP'] = maxHist
    #securities['EURHUF'] = maxHist
    securities['EURJPY'] = maxHist
    securities['EURNOK'] = maxHist
    securities['EURNZD'] = maxHist
    #securities['EURPLN'] = maxHist
    securities['EURSEK'] = maxHist
    securities['GBPCAD'] = maxHist
    securities['GBPCHF'] = maxHist
    securities['GBPJPY'] = maxHist
    securities['GBPUSD'] = maxHist
    securities['NZDCAD'] = maxHist
    securities['NZDJPY'] = maxHist
    securities['NZDUSD'] = maxHist
    securities['USDCAD'] = maxHist
    securities['USDCHF'] = maxHist
    #securities['USDCNY'] = maxHist
    securities['USDCZK'] = maxHist
    #securities['USDHKD'] = maxHist
    securities['USDHUF'] = maxHist
    #securities['USDINR'] = maxHist
    securities['USDJPY'] = maxHist
    securities['USDMXN'] = maxHist
    securities['USDNOK'] = maxHist
    securities['USDPLN'] = maxHist
    securities['USDSEK'] = maxHist
    securities['USDSGD'] = maxHist
    #securities['USDTHB'] = maxHist
    #securities['USDTRY'] = maxHist
    securities['USDZAR'] = maxHist
    securities['XAGUSD'] = maxHist
    securities['XAUUSD'] = maxHist

    def initPortalHeatMap(self):
        heatArray = []
        from pyroTechLibNEW import pyroTechSeries
        pt  = pyroTechSeries()
        for s in self.securities.keys():
            print 'Processing', s
            heatDico = {}
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
            logging.debug('**** FOR ' + heatDico['securityName'] + ' ******\n')
            logging.debug(str(heatDico) + '\n')
            heatArray.append(heatDico)
        return heatArray
     
    def runAllDates(self):
        allSignals = []
        for s in self.securities.keys():
            print 'Running for Security ', s
            heatDico = {}
            pt = acTechSeries(s[:3], s[3:], 360, 'D', 'True')
            heatDico['securityName'] = s
            heatDico['MA1'] = pt.allMean(lag = 21)
            heatDico['MA2'] = pt.allMean(lag = 5)
            heatDico['MACrossover'] = pt.allmaCrossOver(5,21)
            heatDico['ADX'] = pt.allADX()
            heatDico['RSI'] = pt.allRSI()
            heatDico['Stochastics'] = pt.allStochastics()
            heatDico['MACD'] = pt.allMACD()
            heatDico['Doji'] = pt.allDoji()
            heatDico['Engulfing'] = pt.allEngulf()
            heatDico['KRD'] = pt.allKRD()
            heatDico['Hammer/SS'] = pt.allHammer()
            allSignals.append(heatDico)
        return self.getAllScores(allSignals)

    def getAllScores(self, allSignals):
        dateArray = []
        completeMenu = []
        maDates = allSignals[0]['MA1'][0]
        for d in maDates:
            errorFlag = False
            heatArray = []
            print 'Retrieving indications for date ', d
            for s in allSignals:
                heatDico = {}
                try:
                    heatDico['securityName']=s['securityName']
                    ind = maDates.index(d)
                    heatDico['MA1']=s['MA1'][1][ind]
                    ind = s['MA2'][0].index(d)
                    heatDico['MA2']=s['MA2'][1][ind]
                    ind = s['MACrossover'][0].index(d)
                    heatDico['MACrossover']=s['MACrossover'][1][ind]
                    ind = s['ADX'][0].index(d)
                    heatDico['ADX']=s['ADX'][1][ind]
                    ind = s['RSI'][0].index(d)
                    heatDico['RSI']=s['RSI'][1][ind]
                    ind = s['Stochastics'][0].index(d)
                    heatDico['Stochastics']=s['Stochastics'][1][ind]
                    ind = s['MACD'][0].index(d)
                    heatDico['MACD']=s['MACD'][1][ind]
                except:
                    import traceback
                    traceback.print_exc(file=sys.stdout)
                    print 'No consistency in date'
                    errorFlag = True
                    break
                try:
                    ind = s['Doji'][0].index(d)
                    heatDico['Doji']=s['Doji'][1][ind]
                    ind = s['DojiActive'][0].index(d)
                    heatDico['DojiActive']=s['Doji'][2][ind]
                except:
                    heatDico['Doji']=''
                    heatDico['DojiActive']=''
                try:
                    ind = s['Engulfing'][0].index(d)
                    heatDico['Engulfing']=s['Engulfing'][1][ind]
                    ind = s['EngulfingActive'][0].index(d)
                    heatDico['EngulfingActive']=s['Engulfing'][2][ind]
                except:
                    heatDico['Engulfing']=''
                    heatDico['EngulfingActive']=''
                try:
                    ind = s['KRD'][0].index(d)
                    heatDico['KRD']=s['KRD'][1][ind]
                    ind = s['KRDActive'][0].index(d)
                    heatDico['KRDActive']=s['KRD'][2][ind]
                except:
                    heatDico['KRD']=''
                    heatDico['KRDActive']=''
                try:
                    ind = s['Hammer/SS'][0].index(d)
                    heatDico['Hammer/SS']=s['Hammer/SS'][1][ind]
                    ind = s['Hammer/SSActive'][0].index(d)
                    heatDico['Hammer/SSActive']=s['Hammer/SS'][2][ind]
                except:
                    heatDico['Hammer/SS']=''
                    heatDico['Hammer/SSActive']=''
                heatArray.append(heatDico)
            if (errorFlag==False):
                print [k['securityName'] for k in heatArray]
                barMenu, allDrinks = absoluteRank(heatArray)
                completeMenu.append((d, barMenu))
                print 'trades for date ', d, barMenu, allDrinks
        return completeMenu
                
    
if __name__=="__main__":
    print 'logging at ', LOG_FILENAME
    t = StartupHistoric()
    print 'Starting backtest...'
    '''Below line added to attempt a backtest'''
    f=open('backtestresults.txt', 'w')
    f.write(str(t.runAllDates()))
    f.close()
    print 'done.'
    
    


    
