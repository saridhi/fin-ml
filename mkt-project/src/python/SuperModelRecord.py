#!/usr/bin/python2.5
import sys
import os
import random
import logging
import datetime

from addSpreadToSMRecords import Spread

LOG_FILENAME = os.environ['CODE_DIR']+'/logs/backtest-%s.log' %str(datetime.date.today())
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

MAX_SPREAD_PERC = 0.05

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

class OHLC:
    def __init__(self, o, h, l, c):
        self.o = o.rstrip('\r\n')
        self.h = h.rstrip('\r\n')
        self.l = l.rstrip('\r\n')
        self.c = c.rstrip('\r\n')

    def O(self):
        return self.o

    def H(self):
        return self.h

    def L(self):
        return self.l

    def C(self):
        return self.c
    
    def ToString(self):
        return ('%s,%s,%s,%s' %(self.o, self.h, self.c, self.c))

def StringToOHLC(ohlc_string):
    ohlc_parts = ohlc_string.split(",")
    O = ohlc_parts[0]
    H = ohlc_parts[1]
    L = ohlc_parts[2]
    C = ohlc_parts[3]
    if (float(H) < float(L)):
        logging.warning('%s has high < low.' %ohlc_string)
    return OHLC(O, H, L, C)

class SuperModelRecord:
    def __init__(self, pair):
        self.pair = pair
        self.today_ohlc = None
        self.tomorrow_ohlc = None
        self.verdict = ''
        self.score = 0.0
        self.attribs = {}
        self.profit_perc = 0.0
        self.mls = 0
        self.dollars_per_point = 0
        self.sp = Spread()

    def SetAttrib(self, k, v):
        self.attribs[k] = v

    def getSpread(self, pair):
        return self.sp.getSpread(pair)
        

    def SetMls(self, mls):
        self.mls = mls

    def SetDollarsPerPoint(self, dollars_per_point):
        self.dollars_per_point = dollars_per_point

    def SetTodayOHLC(self, today_ohlc):
        self.today_ohlc = today_ohlc

    def SetTomorrowOHLC(self, tomorrow_ohlc):
        self.tomorrow_ohlc = tomorrow_ohlc
    
    def SetNextDate(self, next_date):
        self.next_date = next_date

    def SetVerdict(self, verdict):
        self.verdict = verdict

    def SetScore(self, score):
        self.score = score

    def ToCSVString(self):
        to_str = self.today_ohlc.ToString()
        tm_str = self.tomorrow_ohlc.ToString()
        str = '%s,%s,%s,%s,%.4f,%s,%d,%f' %(self.pair, self.next_date, 
                                         to_str,
                                         tm_str,
                                         self.score,
                                         self.verdict, self.mls, self.dollars_per_point)
        for attrib in ATTRIBS:
            str = str + ',' + self.attribs[attrib]
            
        return str

    def FromCSVString(self, string):
        parts = string.split(',')
        self.pair = parts[0]
        self.next_date = parts[1]
        self.today_price = StringToOHLC(",".join(parts[2:6]))
        self.tomorrow_price = StringToOHLC(",".join(parts[6:10]))
        self.score = float(parts[10])
        self.verdict = parts[11]
        self.mls = int(parts[12])
        self.dollars_per_point = float(parts[13])
        for index in range(len(ATTRIBS)):
            self.attribs[ATTRIBS[index]] = parts[14 + index]

    def SMRHeader(self):
        ret_str = 'Pair,Next_Date,Today.O,Today.H,Today.L,Today.C,' + \
            'Tomorrow.O,Tomorrow.H,Tomorrow.L,Tomorrow.C,' + \
            'Score_Today,Prediction_Today,Mls,DollarsPerPoint'
        for a in ATTRIBS:
            ret_str += ',' + a
        return ret_str

class DataSet:
    def __init__(self, csv_file):
        self.records = []
        self.ReadFile(csv_file)

    def Records(self):
        return self.records

    def ReadFile(self, csv_file):
        if not os.path.exists(csv_file):
            print 'ERROR: %s does not exist.' %csv_file
        f = open(csv_file, 'r')
        first = True
        for row in f:
            if first:
                first = False
                continue
            s = row.rstrip('\n')
            parts = s.split(',')
            smr = SuperModelRecord(parts[0])
            smr.FromCSVString(s)
            self.records.append(smr)
        print 'processed %s with %d records.' %(csv_file, len(self.records))

class Strategy:

    def __init__(self, data_set):
        self.data_set = data_set
        self.profit_percs = None

    def PredictDirection(self, smr):
        if random.randint(0, 1) == 0:
            return ('Fizz', 1.0, -1, -1)
        else:
            return ('Spill', 1.0, -1, -1)

    def NumPredictions(self):
        if self.profit_percs is None:
            print 'ERROR: CalculateProfitAndLoss not called.'
            sys.exit(2)
        return len(self.profit_percs)
        
    def CalculateProfitAndLoss(self, start_amount = 100.0, use_spread = True):
        self.profit_percs = []
        total_amount = start_amount
        count = 0
        for smr in self.data_set.Records():
            next_smr = None
            if count < len(self.data_set.Records()) - 1:
                next_smr = self.data_set.Records()[count + 1]
            count = count + 1
            spread = 0.0
            if use_spread:
                spread = smr.getSpread()

            (dir, conf, profit_limit, stop_loss) = self.PredictDirection(smr)
            if profit_limit == -1 and stop_loss == -1:
                print 'ERROR: set up to one of profit_limit and stop_loss.'
                print 'I cannot figure out which one happened first.'
                sys.exit(2)
            profit_perc = 0.0
            if dir == 'Fizz':
                if profit_limit != -1:
                    hit_profit_limit = False
                    # see if there was a high price beyond the profit_limit
                profit_perc = \
                    (smr.next_price - smr.today_price - spread) / smr.today_price
                self.profit_percs.append(profit_perc)
            elif dir == 'Spill':
                profit_perc = \
                    (smr.today_price - smr.next_price - spread) / smr.today_price
                self.profit_percs.append(profit_perc)
            total_amount = conf * (1.0 + profit_perc) * total_amount + (1 - conf) * total_amount
        return total_amount

class BasicSMStrategy(Strategy):
    def PredictDirection(self, smr):
        if smr.score > 0.0:
            return ('Fizz', smr.score / 10.0, -1, -1)
        elif smr.score < 0.0:
            return ('Spill', -1.0 * smr.score / 10.0, -1, -1)
        return ('Flat', 1.0, -1, -1)


class HighConfidenceStrategy(Strategy):
    def PredictDirection(self, smr):
        if smr.score > 4.0:
            return ('Fizz', 1.0, -1, -1)
        elif smr.score < -4.0:
            return ('Spill', -1.0, -1, -1)
        return ('Flat', 1.0, -1, -1)

class OracleStrategy(Strategy):
    def PredictDirection(self, smr):
        if smr.next_price > smr.today_price:
            return ('Fizz', 1.0, -1, -1)
        else:
            return ('Spill', 1.0, -1, -1)

class NoHighSpreadStrategy(Strategy):

    def __init__(self, data_set):
        self.base_strategy = None
        Strategy.__init__(self, data_set)

    def UseBaseStrategy(self, base_strategy):
        self.base_strategy = base_strategy

    def PredictDirection(self, smr):
        if self.base_strategy is None:
            print 'ERROR: please set base strategy using UseBaseStrategy'
            sys.exit(2)
        if smr.spread * 100.00 / smr.today_price > MAX_SPREAD_PERC:
            logging.warning('High spread. Ignoring ' + smr.ToCSVString())
            return ('Flat', 1.0, -1, -1)
        return self.base_strategy.PredictDirection(smr)


def main():
    args = sys.argv[1:]
    if not args:
        print 'Usage: %s [sm-records filename]' %sys.argv[0]
        sys.exit(2)
    filename = args[0]
    if not os.path.exists(filename):
        print 'ERROR: file %s not found.' %filename
        sys.exit(2)
    ds = DataSet(filename)

    print 'Without Spreads'

    random_strategy = Strategy(ds)
    pnl0 = random_strategy.CalculateProfitAndLoss(use_spread = False)
    print 'random pnl = ', pnl0, '# predictions', random_strategy.NumPredictions()
    
    basic_sm_strategy = BasicSMStrategy(ds)
    pnl2 = basic_sm_strategy.CalculateProfitAndLoss(use_spread = False)
    print 'basic sm pnl = ', pnl2, '# predictions', basic_sm_strategy.NumPredictions()

    print 'With Spreads Now'

    random_strategy = Strategy(ds)
    pnl0 = random_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'random pnl = ', pnl0, '# predictions', random_strategy.NumPredictions()
    
    basic_sm_strategy = BasicSMStrategy(ds)
    pnl2 = basic_sm_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'basic sm pnl = ', pnl2, '# predictions', basic_sm_strategy.NumPredictions()

    high_conf_strategy = HighConfidenceStrategy(ds)
    pnl3 = high_conf_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'high conf pnl = ', pnl3, '# predictions', high_conf_strategy.NumPredictions()

    print 'With Spreads But No High Spreads'

    n_random_strategy = NoHighSpreadStrategy(ds)
    n_random_strategy.UseBaseStrategy(random_strategy)
    pnl0 = n_random_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'random pnl = ', pnl0, '# predictions', n_random_strategy.NumPredictions()
    
    n_basic_sm_strategy = NoHighSpreadStrategy(ds)
    n_basic_sm_strategy.UseBaseStrategy(basic_sm_strategy)
    pnl2 = n_basic_sm_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'basic sm pnl = ', pnl2, '# predictions', n_basic_sm_strategy.NumPredictions()

    n_high_conf_strategy = NoHighSpreadStrategy(ds)
    n_high_conf_strategy.UseBaseStrategy(high_conf_strategy)
    pnl2 = n_high_conf_strategy.CalculateProfitAndLoss(use_spread = True)
    print 'high conf pnl = ', pnl2, '# predictions', n_high_conf_strategy.NumPredictions()

if __name__ == "__main__":
    main()
