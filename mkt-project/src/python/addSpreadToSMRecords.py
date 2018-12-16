import sys
sys.path.append('C:\\Dev')
from techseriesNEW import maxHist, acTechSeries

class Spread():

    securities = {}
    securities['AUDJPY'] = 0.06
    securities['AUDNZD'] = 0.0009
    securities['AUDUSD'] = 0.00027
    securities['CADJPY'] = 0.0525
    securities['CHFJPY'] = 0.042
    securities['EURAUD'] = 0.0006
    securities['EURCAD'] = 0.0006
    securities['EURCHF'] = 0.0003
    securities['EURCZK'] = 0.05
    securities['EURDKK'] = 0.00075
    securities['EURGBP'] = 0.00025
    securities['EURHUF'] = 1.50
    securities['EURJPY'] = 0.0355
    securities['EURNOK'] = 0.0120
    securities['EURPLN'] = 0.0225
    securities['EURSEK'] = 0.0125
    securities['EURTRY'] = 0.018
    securities['EURUSD'] = 0.000135
    securities['GBPCHF'] = 0.0006
    securities['GBPJPY'] = 0.0675
    securities['GBPUSD'] = 0.000375
    securities['NZDUSD'] = 0.0005
    securities['USDCAD'] = 0.0004
    securities['USDCHF'] = 0.000375
    securities['USDCNY'] = 0.0180
    securities['USDDKK'] = 0.0015
    securities['USDHKD'] = 0.0075
    securities['USDINR'] = 0.105
    securities['USDJPY'] = 0.024
    securities['USDMXN'] = 0.0100
    securities['USDNOK'] = 0.0120
    securities['USDPLN'] = 0.0225
    securities['USDSAR'] = 0.0075
    securities['USDSGD'] = 0.000675
    securities['USDTHB'] = 1.0
    securities['USDTRY'] = 0.0100
    securities['USDTWD'] = 0.15
    securities['USDZAR'] = 0.02
    securities['XAGUSD'] = 0.03
    securities['XAUUSD'] = 0.50
    securities['AUDCAD'] = 0.0005
    securities['EURNZD'] = 0.0012
    securities['GBPCAD'] = 0.0009
    securities['NZDCAD'] = 0.0008
    securities['NZDJPY'] = 0.06
    securities['USDCZK'] = 0.05
    securities['USDHUF'] = 2.5
    securities['USDSEK'] = 0.0125
    # non-oegg pairs
    securities['GBPNZD'] = 0.0004
    securities['AUDCHF'] = 0.0004
    securities['GBPAUD'] = 0.0004
    default_spread = 0.1

    __shared_state = {}
    spreadRatios = {}

    #Singleton pattern
    def __init__(self):
        self.__dict__ = self.__shared_state
    
    def getSpread(self, symbol):
        try:
            return self.securities[symbol]
        except:
             print 'Security ', symbol, ' has no spread.'
             return self.default_spread
        
    def getSpreadRatios(self):
        if len(self.spreadRatios) == 0:
            from PairConverter import PairConverter
            p = PairConverter('dummy')
            for s in self.securities.keys():
                try:
                    ts_b = acTechSeries(s[:3], s[3:], 360, 'D', 'True')
                    spreadRatio = self.securities[s]/(p.expMean(ts_b.trueRange(),60))
                except:
                    continue
                self.spreadRatios[s]=spreadRatio
        return self.spreadRatios


def readFile(fileName="sm_records.csv", 
             out_fileName="sm_records_with_spread.csv"):
    f = open(fileName, 'r')
    f2 = open(out_fileName, 'w')
    line = f.readline()
    f2.write(line.replace('\n', ',Spread\n'))
    sp = Spread()
    while 1:
        line = f.readline()
        if not line:
            break
        components = line.split(',')
        spread = sp.getSpread(components[0])
        newline = line.rstrip('\n') + str(spread) + '\n'
        f2.write(newline)


if __name__=="__main__":
    None
    #readFile("sm_records.csv", "sm_records_with_spread.csv")
