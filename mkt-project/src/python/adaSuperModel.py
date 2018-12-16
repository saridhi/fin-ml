import os
import commands
import smtplib
import email.utils
import getpass
import sys
import datetime
import time
import random

from randomSong import RandomSong
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from techseriesNEW import maxHist, acTechSeries
from PairConverter import PairConverter
from heatmap import heatMapRank, toprank, bottomrank, getColorDico, selection_sort, absoluteRank

import glob
import logging
import logging.handlers

if not 'LOG_DIR' in os.environ:
	print 'ERROR: LOG_DIR is not an environment variable.'
	sys.exit(2)


LOG_FILENAME = '%s/adaSuperModel-%s.log' %(os.environ['LOG_DIR'], str(datetime.date.today()))

my_logger = logging.getLogger('adaSuperModel.py')
my_logger.setLevel(logging.DEBUG)
hdlr = logging.FileHandler(LOG_FILENAME)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
my_logger.addHandler(hdlr)

UPTICK_MAX=0.015

HTML_TEMPLATE = "<html>\n" + \
"<head>\n" + \
"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">\n" + \
"<base target=\"_top\">\n" + \
"<title>ADA Drink Whiz</title>\n" + \
"</head>\n" + \
"<body>\n" + \
"<span style=FONT-FAMILY:Helvetica>\n" + \
"<div>\n" + \
"  Hi ADA Lords,\n" + \
"</div>\n" + \
"<div>\n" + \
"  <br>\n" + \
"</div>\n" + \
"<div>\n" + \
"  Here are my suggestions with drink sizes and conflict-free drinking advice.\n" + \
"</div>\n" + \
"<div style=TEXT-ALIGN:left>\n" + \
"  <br>\n" + \
"</div>\n" + \
"</span>\n" + \
"<div>\n" + \
"  <div>\n" + \
"    <table bgcolor=#f3f3f3 border=0 bordercolor=#000000 cellpadding=5 cellspacing=3 class=zeroBorder id=afgx style=TEXT-ALIGN:left>\n" + \
"      <tbody>\n" + \
"      <tr>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=COLOR:#274e13><b>Fizziest</b></span>\n" + \
"        </td>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <b><span style=COLOR:#ff0000>Spilliest</span></b>\n" + \
"        </td>\n" + \
"      </tr>\n" + \
"      <tr>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#274e13>FIZZ1<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#ff0000>SPILL1<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"      </tr>\n" + \
"      <tr>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#274e13>FIZZ2<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#ff0000>SPILL2<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"      </tr>\n" + \
"      <tr>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#274e13>FIZZ3<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#ff0000>SPILL3<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"      </tr>\n" + \
"      <tr>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#274e13>FIZZ4<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"        <td style=TEXT-ALIGN:center width=50%>\n" + \
"          <span style=\" FONT-FAMILY:Helvetica\"><span style=COLOR:#ff0000>SPILL4<br>\n" + \
"          </span></span>\n" + \
"        </td>\n" + \
"      </tr>\n" + \
"      </tbody>\n" + \
"    </table>\n" + \
"  </div>\n" + \
"  <span style=FONT-FAMILY:Helvetica>\n" + \
"  <div style=TEXT-ALIGN:left>\n" + \
"    <br>\n" + \
"  </div>\n" + \
"  <div style=TEXT-ALIGN:left>\n" + \
"    Drink wisely,<br>\n" + \
"  <br></div>\n" + \
"  <div style=TEXT-ALIGN:left>\n" + \
"    The ADA SuperModel SuperBrain (" + os.environ['USER'] + ").\n" + \
"  </div>\n" + \
"  <div style=TEXT-ALIGN:left>\n" + \
"    <hr>\n" + \
"  </div>\n" + \
"  <br>\n" + \
"  FORTUNE_COOKIE\n" + \
"  </span>\n" + \
"  <br>\n" + \
"</div>\n" + \
"<br><hr>\n" + \
"\n" + \
"DEBUG_STRING" + \
"<br><hr></body>\n" + \
"</html>\n"

#Load up the maximum possible history for efficient caching

class Startup():
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

    '''Ideal Spread is the maximum spread you are willing (being lenient) to bear for the ideal Security'''
    def removeTharras(self):
        from addSpreadToSMRecords import Spread
        poppedDrinks = []
	poppedRatios = []
        p = PairConverter('dummy')
        spread = Spread()
        idealSecurity = 'EURUSD'
        #idealSpread = Spread.securities[idealSecurity]
        idealSpread = .0015
        ts_a = acTechSeries(idealSecurity[:3], idealSecurity[3:], 360, 'D', 'True')
        idealSpreadRatio = idealSpread/(p.expMean(ts_a.trueRange(),60))      
        spreadRatios = spread.getSpreadRatios()
        for s in self.securities.keys():
            '''exception for medals and spoons'''
            if s in ('XAUUSD', 'XAGUSD'):
                continue
            ts_b = acTechSeries(s[:3], s[3:], 360, 'D', 'True')
            spreadRatio = spreadRatios[s]
            if spreadRatio > idealSpreadRatio:
                self.securities.pop(s)
                poppedDrinks.append(s)
		poppedRatios.append(spreadRatio)
        return (poppedDrinks, poppedRatios)

    def dailyChanges(self, entries = 4):
        changesArray = []
        for s in self.securities.keys():
            t = acTechSeries(s[0:3],s[-3:], self.securities[s], 'D')
            changesArray.append((s,(t.securityC().Values()[-1]-t.securityC().Values()[-2])/t.securityC().Values()[-2]))

        sortedList = selection_sort(changesArray)
        topentries = []
        bottomentries = []          
        return toprank(sortedList),bottomrank(sortedList)
            
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
            my_logger.debug(str(heatDico))
            heatArray.append(heatDico)
        return heatArray

def send_mail(to_guy, to_addr, text, html, subject):
            # Prompt the user for connection info
            servername = 'smtp.gmail.com'
            username = sys.argv[1]
            password = sys.argv[2]

            # Create the message
            msg = MIMEMultipart('alternative')
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')
            msg.attach(part1)
            msg.attach(part2)
            msg.set_unixfrom('SuperModel')
            msg['To'] = email.utils.formataddr((to_guy, to_addr))
            msg['From'] = email.utils.formataddr(('SuperModel', 'dhiren.sarin@gmail.com'))
            msg['Subject'] = subject

            server = smtplib.SMTP(servername)
            try:
                server.set_debuglevel(False)
                
                # identify ourselves, prompting server for supported features
                server.ehlo()

                # If we can encrypt this session, do it
                if server.has_extn('STARTTLS'):
                    server.starttls()
                    server.ehlo() # re-identify ourselves over TLS connection

                server.login(username, password)
                server.sendmail('dhiren.sarin@gmail.com', [to_addr], msg.as_string())
            finally:
                server.quit()

def GetFortune():
    fortune = 'I am bored today.'
    if os.path.exists('/sw/bin/fortune'):
        fortune = commands.getoutput('/sw/bin/fortune')
    return fortune

def ToBeer(code):
    if code == 'MXN': return 'Corona'
    if code == 'SGD': return 'Tiger'
    if code == 'USD': return 'Bud'
    if code == 'EUR': return 'Stella'
    if code == 'AUD': return 'Fosters'
    if code == 'NZD': return 'Penguin'
    if code == 'GBP': return 'Pale Ale'
    if code == 'NOK': return 'Nokia'
    if code == 'JPY': return 'Sapporo'
    if code == 'INR': return 'Kingfisher'
    if code == 'CHF': return 'Nestle'
    if code == 'CAD': return 'Molson'
    if code == 'TRY': return 'Efes'
    if code == 'HUF': return 'SABMiller'
    if code == 'XAU': return 'Medal'
    if code == 'XAG': return 'Spoon'
    if code == 'SEK': return 'Spendrups'
    if code == 'PLN': return 'Zweic'
    if code == 'PLZ': return 'Zweic'
    if code == 'CNY': return 'Tsing Tao'
    return code

def ToString(entries):
    r = ''
    try:
        for e in entries:
            if e[0]:
                r = r + ToBeer(e[0][:3]) + '/' + ToBeer(e[0][3:]) + ' : ' + str(e[1]) + '\n'
    except:
        return str(entries)
    return r

def ToStringTharra(entries, scores):
    r = ''
    try:
        for i in range(len(entries)):
		e = entries[i]
		score = scores[i]
		r = r + ToBeer(e[:3]) + '/' + ToBeer(e[3:]) + ' (' + str(int(score*10000) / 100.00).rstrip('0') + '), '
	r = r[:-2]
    except:
        return str(entries)
    return r

def LastPrice(pair, dir='FIZZ'):
    data_dir = os.environ['DATA_DIR']
    f = os.path.join(data_dir, '$' + pair + '.txt')
    if not os.path.exists(f):
        print 'ERROR: %s does not exist.' %f
        sys.exit(1)
    fi = open(f, 'r')
    last = ''
    for row in fi:
        last = row
    el = last.split(',')
    last_price = float(el[4].rstrip('\n').rstrip('0'))
    next_price = last_price
    if dir == 'FIZZ':
        next_price = last_price * (1 + UPTICK_MAX)
    else:
        if dir == 'SPILL':
            next_price = last_price * (1 - UPTICK_MAX)
    st = str(last_price).rstrip('0') + ' on ' + el[0][:-5] + ' going for ' + \
        str(next_price).rstrip('0')
    return st

def ToStringHTML(entries, html_string, repl_str):
    count = 1
    p = PairConverter('dummy')
    for e in entries:
        repl = repl_str + str(count)
        if e[0]:
            dir = repl_str
            new_string = \
                ToBeer(e[0][:3]) + '/' + ToBeer(e[0][3:]) + \
                ' : ' + str(e[1]) + ' (' + LastPrice(e[0], dir) + ' Mls: '+ str(int(p.getAbsoluteMls(e[0])))+')'
            html_string = html_string.replace(repl, new_string)
            
        else:
            html_string = html_string.replace(repl, '')
        count = count + 1
    return html_string

def DictToStr(dic):
    str = ''
    for k in dic:
        if k == 'securityName':
            continue
        if dic[k]:
            str += k + ': ' + dic[k] + ', '
        else:
            str += k + ': ?, '
    str = str[:-2]
    return str

def ConstructDebugString(entries, map):
    ret_str = ''
    for entry in entries:
        if entry and entry[0]:
            for dico in map:
                if dico['securityName'] == entry[0]:
                    ret_str += '<b>' + ToBeer(entry[0][:3]) + '/' + ToBeer(entry[0][3:]) + '</b>: ' + DictToStr(dico) + '<br>'
                    break
    return ret_str

def GetDrinkSize(barMenu, drinkScores):
    pairConverter = PairConverter('dummy')
    absoluteMls = 0
    if len(barMenu) == 0:
        return 0
    elif len(barMenu) == 1:
        return pairConverter.getAbsoluteMls(barMenu[0])*2
    else:
        pairRatio = pairConverter.pairRatio(barMenu[0], barMenu[1])
        secondDrink = barMenu[1]
        firstDrink = barMenu[0]
        absoluteMls = pairConverter.getAbsoluteMls(secondDrink)
        adjustedSize = pairConverter.scaleByCorrelation(firstDrink, secondDrink)
	scaledFactor1 = pairConverter.scaleByScore(drinkScores[0])
	scaledFactor2 = pairConverter.scaleByScore(drinkScores[1])
        return (int(adjustedSize*pairRatio*absoluteMls) * scaledFactor1, int(adjustedSize*absoluteMls) * scaledFactor2)

def AdaSMMain(heatMap):
    topentries, bottomentries = heatMapRank(4, heatMap)
    my_logger.info('Top ' + str(topentries))
    my_logger.info('Bottom ' + str(bottomentries))
    barMenu, allDrinks = absoluteRank(heatMap)
    return (topentries, bottomentries, barMenu, allDrinks)

def adaModel():
    t = Startup()
    (poppedDrinks, poppedRatios) = t.removeTharras()
    my_logger.debug('Popped ' + str(poppedDrinks))
    heatMap = t.initPortalHeatMap()
    (topentries, bottomentries, barMenu, allDrinks) = AdaSMMain(heatMap)
    print barMenu, ' Bar Menu'
    logging.debug('Bar Menu ' + str(barMenu))
    drinkSizes = []
    try:
        drinkArray = [f[0] for f in barMenu]
        drinkSizes = GetDrinkSize(drinkArray, [f[1] for f in barMenu])
    except:
        drinkSizes = 'Could not size the drinks (incompetent bartender)'

    htmlString = HTML_TEMPLATE
    htmlString = ToStringHTML(topentries, htmlString, 'FIZZ')
    htmlString = ToStringHTML(bottomentries, htmlString, 'SPILL')
    debugString = '<br><b>Top Two Drinks</b>:<br>' + ToString(barMenu) + '<br>'
    debugString += '<br><b>Respective drink sizes</b>:<br>' + str(drinkSizes) + '<br>'
    debugString += '<br><b>Entire Menu</b>:<br>' + ToString(allDrinks) + '<br>'
    debugString += '<br><b>Tharra drinks with too much chaudha</b>:<br>' + ToStringTharra(poppedDrinks, poppedRatios) + '<br>'
    htmlString = htmlString.replace('DEBUG_STRING', debugString)
    song_path = '/Users/%s/Documents/Sing\ that\ iTune\!' %os.environ['USER']
    if random.randint(0, 1) and os.path.exists(song_path):
        rs = RandomSong(song_path)
        song, lyrics, band = rs.RandomSongWithLyrics()
        some_song = \
            'I have cool taste. Did you try <b>' + song + '</b> by ' + \
            '<b>' + band + '</b>?' + \
            '<br><br>' + \
            lyrics
        htmlString = htmlString.replace('FORTUNE_COOKIE', some_song)
    else:
        htmlString = htmlString.replace('FORTUNE_COOKIE', GetFortune())

    plain_string = 'Fizziest: \n'
    plain_string += ToString(topentries)
    plain_string += '\nSpillest: \n'
    plain_string += ToString(bottomentries)

    print plain_string
    print 'done', datetime.datetime.today()
    subject = 'Supermodel suggestions on ' + str(datetime.date.today())
    send_mail('Dhiren Sarin', 'dhiren.sarin@gmail.com', plain_string, htmlString, subject)
     
if __name__=="__main__":
    #Test drink sizing
    ''' print GetDrinkSize(['XAUUSD', 'USDSGD']) '''
    if len(sys.argv) == 1:
        print 'Usage: %s [gmail-addr] [gmail-password]' %sys.argv[0]
        sys.exit(1)
    adaModel()
