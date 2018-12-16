import sys
#import logging

import datetime
from techseries import maxHist, acTechSeries
#from fxstdtimeserie import liborTS

import time
#import logx

#logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


#Load up the maximum possible history for efficient caching
class Startup():

    securities = {}
    today = datetime.date.today()
    #maxHist = (today - datetime.date(1992,5,18)).days
    #maxHist = 1000
    hammerHist = None
    engulfHist = None
    krdHist = None
    dojiHist = None
    
    
    '''securities['EURUSD']=(today - datetime.date(1992,5,18)).days
    securities['GBPUSD']=(today - datetime.date(1992,5,18)).days
    securities['CADJPY']=(today - datetime.date(1992,5,18)).days
    securities['USDCHF']=(today - datetime.date(1992,5,18)).days
    securities['USDJPY']=(today - datetime.date(1992,5,18)).days
    securities['NZDUSD']=(today - datetime.date(1992,5,18)).days
    securities['AUDUSD']=(today - datetime.date(1992,5,18)).days
    securities['USDCAD']=(today - datetime.date(1992,5,18)).days
    securities['EURGBP']=(today - datetime.date(1995,7,18)).days
    securities['EURCHF']=(today - datetime.date(1997,1,24)).days
    securities['EURJPY']=(today - datetime.date(1995,7,19)).days
    securities['EURAUD']=(today - datetime.date(1999,1,5)).days
    securities['EURNZD']=(today - datetime.date(1992,5,19)).days
    securities['GBPJPY']=(today - datetime.date(1993,7,14)).days
    securities['AUDJPY']=(today - datetime.date(1992,6,22)).days
    securities['NZDJPY']=(today - datetime.date(1992,5,18)).days
    securities['CHFJPY']=(today - datetime.date(1996,7,26)).days'''

    securities['EURUSD']=maxHist
    securities['GBPUSD']=maxHist
    securities['CADJPY']=maxHist
    securities['USDCHF']=maxHist
    securities['USDJPY']=maxHist
    securities['NZDUSD']=maxHist
    securities['AUDUSD']=maxHist
    securities['USDCAD']=maxHist
    securities['EURGBP']=maxHist
    securities['EURCHF']=maxHist
    securities['EURJPY']=maxHist
    securities['EURAUD']=maxHist
    securities['EURNZD']=maxHist
    securities['GBPJPY']=maxHist
    securities['AUDJPY']=maxHist
    securities['NZDJPY']=maxHist
    securities['CHFJPY']=maxHist
    securities['GBPAUD']=maxHist
    securities['GBPCAD']=maxHist
    securities['GBPCHF']=maxHist
    securities['GBPNZD']=maxHist
    securities['AUDNZD']=maxHist
    securities['AUDCAD']=maxHist
    securities['AUDCHF']=maxHist
    securities['USDKRW']=maxHist
    securities['GBPKRW']=maxHist
    securities['EURKRW']=maxHist
    securities['KRWJPY']=maxHist
    securities['USDSGD']=maxHist
    securities['GBPSGD']=maxHist
    securities['EURSGD']=maxHist
    securities['SGDJPY']=maxHist
    securities['USDINR']=maxHist
    securities['GBPINR']=maxHist
    securities['EURINR']=maxHist
    securities['INRJPY']=maxHist
    securities['USDBRL']=maxHist
    securities['EURBRL']=maxHist
    securities['GBPBRL']=maxHist
    securities['BRLJPY']=maxHist
    securities['USDMXN']=maxHist
    securities['EURMXN']=maxHist
    securities['GBPMXN']=maxHist
    securities['MXNJPY']=maxHist
    securities['USDCLP']=maxHist
    securities['EURCLP']=maxHist
    securities['GBPCLP']=maxHist
    securities['USDPLN']=maxHist
    securities['EURPLN']=maxHist
    securities['GBPPLN']=maxHist
    securities['EURHUF']=maxHist
    securities['GBPHUF']=maxHist
    securities['USDHUF']=maxHist
    securities['HUFPLN']=maxHist
    securities['USDZAR']=maxHist
    securities['EURZAR']=maxHist
    securities['GBPZAR']=maxHist
    securities['USDTRY']=maxHist
    securities['EURTRY']=maxHist
    securities['GBPTRY']=maxHist

        
    def initSecurities(self):
        total_tlength = 0.
        for s in self.securities.keys():
            #Init Daily
            t = fxtechseries(s[0:3],s[-3:], self.securities[s], 'D')
            t_start = time.time()
            t.RawData()
            t.isUptrend()
            t.isDowntrend()
            #t.rRegSlope(t.securityC())
            t_end = time.time()
            tlength = t_end-t_start
            total_tlength += tlength
            ptime = str(tlength).split('.')[0]+'.'+str(tlength).split('.')[1][0:2]
            logx.Write('did ; '+s+', '+str(maxHist)+', in '+ptime+' seconds')
            ptotaltime = str(total_tlength).split('.')[0]+'.'+str(total_tlength).split('.')[1][0:2]
        logx.Write('Did the full list in ; '+ptotaltime+' seconds')
        Libors = liborTS('USD','3m',maxHist)
            
    def initPortalHeatMap(self):
        heatArray = []
        from pyroTechLib import pyroTechSeries
        pt  = pyroTechSeries()
        for s in self.securities.keys():
            heatDico = {}
            t = fxtechseries(s[0:3],s[-3:], self.securities[s], 'D')
            heatDico['securityName'] = s
            heatDico['MA1'] = pt.snapMA(asset = s[0:3], base = s[-3:], lag = 21)['Values']
            heatDico['MA2'] = pt.snapMA(asset = s[0:3], base = s[-3:], lag = 5)['Values']
            heatDico['MACrossover'] = pt.snapCrossOver(asset = s[0:3], base = s[-3:])['Values']
            heatDico['ADX'] = pt.snapADX(asset = s[0:3], base = s[-3:])['Values']
            heatDico['RSI'] = pt.snapRSI(asset = s[0:3], base = s[-3:])['Values']
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
         

    def runStats(self):
        asset = ''
        base = ''
        from pyroTechLib import pyroTechSeries
        p = pyroTechSeries()
        for s in self.securities.keys():
            base = base + s[-3:]+','
            asset = asset + s[0:3]+','
        self.hammerHist = p.backtestPortfolio(asset[0:-1],base[0:-1], maxHist,'D','200days','bullHammer,bearShootingStar','80,80', '9,9','stoploss','',True)
        self.dojiHist = p.backtestPortfolio(asset[0:-1],base[0:-1], maxHist,'D','200days','bullDoji,bearDoji','80,80', '9,9','stoploss','',True)
        self.krdHist = p.backtestPortfolio(asset[0:-1],base[0:-1], maxHist,'D','200days','bullKRD,bearKRD','80,80', '9,9','stoploss','',True)
        self.engulfHist = p.backtestPortfolio(asset[0:-1],base[0:-1], maxHist,'D','200days','bullEngulf,bearEngulf','80,80', '9,9','stoploss','',True)

                            
def main():
        
        t = Startup()
        print 'start', datetime.datetime.today()
        print t.initSecurities()
        #print heatMapRank(1, t.initPortalHeatMap())
        #print t.runStats()
        print 'done', datetime.datetime.today()


if __name__=="__main__":
	main()











