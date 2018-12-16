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
from SuperModelRecord import OHLC, SuperModelRecord
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from techseriesNEW import maxHist, acTechSeries
from heatmap import heatMapRank, toprank, bottomrank, getColorDico, selection_sort        
from PairConverter import PairConverter
from adaSuperModel import AdaSMMain


ATTRIBS = ['MA1', 
           'Stochastics', 
           'MA2', 
           'Doji', 
           'RSI', 
           'Hammer/SSActive', 
           'ADX', 'MACD', 
           'Engulfing', 
           'DojiActive', 
           'KRDActive', 
           'MACrossover', 
           'EngulfingActive', 
           'KRD']

def FillSMR(smr, h):
    for a in ATTRIBS:
        smr.SetAttrib(a, h[a])

def ToMMDDYYYY(s):
    sp = s.split('/')
    if not len(sp) == 3:
        print 'Bad date:', s
        sys.exit(1)
    return sp[1] + '/' + sp[0] + '/' + sp[2]

def StringPartsToOHLC(ohlc_parts):
    O = ohlc_parts[0]
    H = ohlc_parts[1]
    L = ohlc_parts[2]
    C = ohlc_parts[3]
    if (float(H) < float(L)):
        logging.warning('%s has high < low.' %ohlc_string)
    return OHLC(O, H, L, C)

class StartupHistoric():
    securities = {}
    securities['AUDCAD'] = maxHist
    securities['AUDJPY'] = maxHist
    securities['AUDNZD'] = maxHist
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
    securities['XAUUSD'] = maxHist

    def initPortalHeatMap(self):
        heatArray = []
        from pyroTechLibNEW import pyroTechSeries
        pt  = pyroTechSeries()
        for s in self.securities.keys():
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
     
if __name__=="__main__":
    print 'logging at ', LOG_FILENAME
    from adaSuperModel import GetDrinkSize, Startup
    t = Startup()
    data_dir = os.environ['DATA_DIR']
    file = open(os.path.join(data_dir, '$USDJPY.txt'), 'r')
    all_dates = []
    first = True
    for row in file:
        all_dates.append(row.split(',')[0].rstrip('\n'))
    start_ = len(all_dates) / 2
    end_ = start_ + 150
    for i in range(start_, end_):
        logging.debug('Pulling last %d day\'s worth of data' %i)
        i_date = all_dates[i]
        print 'looking at SM on ', i_date
        ndp = {}
        tdp = {}
        ndp_date = {}
        incomplete = False
        # Only keep lines 1 through i in all data files and then run.
        for curr in t.securities.keys():
            currency = '$' + curr + '.txt'
            # first copy the file somewhere
            if first:
                shutil.copyfile(os.path.join(data_dir, currency), 
                                os.path.join('/tmp', currency))
            currency_filename = os.path.join('/tmp', '\\' + currency)
            posn_date_cmd = "egrep -n '^" + i_date + "' " + \
                currency_filename + \
                "| awk -F':' '{ print $1 }'"
            (st, ou) = commands.getstatusoutput(posn_date_cmd)
            if not ou:
                print 'command not run properly:', posn_date_cmd
                incomplete = True
                continue
            li_no = int(ou.rstrip('\r\n'))
            # copy prices upto the date in li_no
            cmd = 'head -n %d %s > %s' \
                %(li_no, currency_filename, 
                  os.path.join(data_dir, '\\' + currency))
            (status, output) = commands.getstatusoutput(cmd)
            logging.debug('extracting the first %d entries in %s using command:\n%s' %(i, currency, cmd))
            if not status == 0:
                print 'ERROR: could not execute %s' %cmd
                print output
                sys.exit(1)
            # look in /tmp/$USDJPY.txt for the last two lines
            ndp_cmd = \
                'head -n %d %s | tail -n 2' %(li_no + 1, 
                                              currency_filename)
            (status, output) = commands.getstatusoutput(ndp_cmd)
            logging.debug('getting next day price with %s' %ndp_cmd)
            if not status == 0:
                print 'ERROR: could not execute %s' %cmd
                print output
                sys.exit(1)
            op = output.split('\n')
            ndp[currency[1:-4]] = op[1].rstrip('\n').split(',')[1:]
            tdp[currency[1:-4]] = op[0].rstrip('\n').split(',')[1:]
            ndp_date[currency[1:-4]] = op[1].rstrip('\n').split(',')[0]

        first = False

        if not incomplete:
            print 'Running model for iteration ', i
            (poppedDrinks, poppedRatios) = t.removeTharras()
            logging.debug('Popped ' + str(poppedDrinks))
            heatMap = t.initPortalHeatMap()
            (topentries, bottomentries, barMenu, allDrinks) = AdaSMMain(heatMap)
            p = PairConverter('dummy')

            drinkSizes = []
            try:
                drinkArray = [f[0] for f in barMenu]
                drinkSizes = GetDrinkSize(drinkArray, [f[1] for f in barMenu])
                if len(drinkSizes)<2:
                    continue
            except:
                drinkSizes = 'Could not size the drinks (incompetent bartender)'
                continue
        
            '''for el in topentries:
                sec = el[0]
                if not sec: continue
                smr = SuperModelRecord(sec)
                if not os.path.exists('sm_backtest.csv'):
                    smr_file = open('sm_backtest.csv', 'w')
                    smr_file.write(smr.SMRHeader() + '\n')
                    smr_file.close()

                smr.SetMls(int(p.getAbsoluteMls(sec)))
                smr.SetDollarsPerPoint(p.getDollarsPerPoint(sec))
                smr.SetTodayOHLC(StringPartsToOHLC(tdp[sec]))
                smr.SetTomorrowOHLC(StringPartsToOHLC(ndp[sec]))
                smr.SetNextDate(ToMMDDYYYY(ndp_date[sec]))
                smr.SetVerdict('Fizz')
                smr.SetScore(el[1])
                for h in heatMap:
                    if h['securityName'] == sec:
                        FillSMR(smr, h)

                smr_file = open('sm_backtest.csv', 'a')
                smr_file.write(smr.ToCSVString() + '\n')
                smr_file.close()

            for el in bottomentries:
                sec = el[0]
                if not sec: continue
                smr = SuperModelRecord(sec)
                if not os.path.exists('sm_backtest.csv'):
                    smr_file = open('sm_backtest.csv', 'w')
                    smr_file.write(smr.SMRHeader() + '\n')
                    smr_file.close()

                smr.SetMls(int(p.getAbsoluteMls(sec)))
                smr.SetDollarsPerPoint(p.getDollarsPerPoint(sec))
                smr.SetTodayOHLC(StringPartsToOHLC(tdp[sec]))
                smr.SetTomorrowOHLC(StringPartsToOHLC(ndp[sec]))
                smr.SetNextDate(ToMMDDYYYY(ndp_date[sec]))
                smr.SetVerdict('Spill')
                smr.SetScore(el[1])
                for h in heatMap:
                    if h['securityName'] == sec:
                        FillSMR(smr, h)

                smr_file = open('sm_backtest.csv', 'a')
                smr_file.write(smr.ToCSVString() + '\n')
                smr_file.close()'''

            for el, sz in zip(barMenu, drinkSizes):
                sec = el[0]
                if not sec: continue
                smr = SuperModelRecord(sec)
                if not os.path.exists('sm_backtest.csv'):
                    smr_file = open('sm_backtest.csv', 'w')
                    smr_file.write(smr.SMRHeader() + '\n')
                    smr_file.close()

                smr.SetMls(int(sz))
                smr.SetDollarsPerPoint(p.getDollarsPerPoint(sec))
                smr.SetTodayOHLC(StringPartsToOHLC(tdp[sec]))
                smr.SetTomorrowOHLC(StringPartsToOHLC(ndp[sec]))
                smr.SetNextDate(ToMMDDYYYY(ndp_date[sec]))
                if el[1]>0:
                    smr.SetVerdict('Fizz')
                elif el[1]<0:
                    smr.SetVerdict('Spill')
                else:
                    smr.SetVerdict('Wine')
                smr.SetScore(el[1])
                for h in heatMap:
                    if h['securityName'] == sec:
                        FillSMR(smr, h)

                smr_file = open('sm_backtest.csv', 'a')
                smr_file.write(smr.ToCSVString() + '\n')
                smr_file.close()
    

    for curr in t.securities.keys():
        currency = '$' + curr + '.txt'
        shutil.copyfile(os.path.join('/tmp', currency),
                        os.path.join(data_dir, currency))

    
