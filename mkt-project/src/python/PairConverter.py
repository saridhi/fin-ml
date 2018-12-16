#!/usr/bin/python2.5
import os
import sys
from techseriesNEW import acTechSeries
from numpy import corrcoef

'''This class is generally used for drink sizing'''
class PairConverter:

    def __init__(self, pair_file): 
        '''self.conversions = self.LoadConversions(pair_file)'''
    
    def Convert(self, pair_a, pair_b):
        pair_a_rate = self.conversions[pair_a]
        pair_b_rate = self.conversions[pair_b]
        print pair_a, pair_a_rate * 0.01, pair_b, pair_b_rate * 0.01

    #if 10000 units invested in pair2, pair1 deserves 10000*ratio
    # This is based on volatily/average range of the last 20 days 
    def pairRatio(self, pair_a, pair_b):
        ts_a = acTechSeries(pair_a[:3], pair_a[3:], 360, 'D', 'True')
        ts_b = acTechSeries(pair_b[:3], pair_b[3:], 360, 'D', 'True')
        trueRangeA = self.expMean(ts_a.trueRange(),20)
        trueRangeB = self.expMean(ts_b.trueRange(),20)
        ratio = (trueRangeB*self.getDollarsPerPoint(pair_b))/(trueRangeA*self.getDollarsPerPoint(pair_a))
        return ratio

    #TODO: right now defaulted to 5000. Should ideally depend on trueRange and loss tolerance, needs improvement
    def getAbsoluteMls(self, pair):
        fundValue = 1350
        lossTolerance = 0.058 #5.8% of fund - max loss per drink -- wee bit high?
        maxDrinkLoss = fundValue * lossTolerance
        ts_a = acTechSeries(pair[:3], pair[3:], 360, 'D', 'True')
        mlsToDrink = maxDrinkLoss/(self.expMean(ts_a.trueRange(),20)*self.getDollarsPerPoint(pair))
        return mlsToDrink

    '''Reduce position sizes to 'scaleDownFactor' of original if r squared is greater than equal to 0.85.
       Initial scaling down before anything else'''
    def scaleByCorrelation(self, pair_a, pair_b):
       scaleDownFactor = 0.85 
       extremeScaleDown = 0.75
       correlationPeriod = 60 #Period within which to check out the correlation
       ts_a = acTechSeries(pair_a[:3], pair_a[3:], 360, 'D', 'True')
       ts_b = acTechSeries(pair_b[:3], pair_b[3:], 360, 'D', 'True')
       coefMatrix = corrcoef(ts_a.securityC().Values()[-correlationPeriod:], ts_b.securityC().Values()[-correlationPeriod:])
       rSquared = coefMatrix[0][1]*coefMatrix[0][1]
       if rSquared>=0.95:
           return extremeScaleDown
       elif rSquared>0.85:
           return scaleDownFactor
       else:
           return 1

    '''Change position weighting according to score, assuming a highest possible score'''
    def scaleByScore(self, score):
         highestScore = 6.5
         factor = abs(score/highestScore)
         if factor > 1.0: #Cap it to 1.0 in case score > highestScore
             factor = 1.0
         scaledFactor = factor*(2-factor)
         return scaledFactor

    #Dollars earned/lost for every point when 10000 units are invested
    #This equates to the current exchange rate for all the pairs against the USD
    #For example 'EURCHF' = 0.87 because 'CHFUSD' is currently 0.87
    def getDollarsPerPoint(self, pair):
        base = pair[3:]
        asset = pair[:3]
        dollarsPerPoint = 0
        if base == 'USD':
            return 1
        elif asset == 'XAU':
            return 100
        else:
            try:
                ts = acTechSeries('USD', pair[3:], 20, 'D', 'True')
                dollarsPerPoint = 1/ts.securityC().Values()[-1]
            except:
                ts = acTechSeries(pair[3:], 'USD', 20, 'D', 'True')
                dollarsPerPoint = ts.securityC().Values()[-1]
        return dollarsPerPoint

    def LoadConversions(self, pair_file):
        ans = {}
        if not os.path.exists(pair_file):
            print 'ERROR: %s does not exist.' %pair_file
            sys.exit(2)

        pairs = open(pair_file, 'r')
        for p in pairs:
            p_filename = os.path.join(os.environ['DATA_DIR'], 
                                      '$' + p.rstrip('\n') + '.txt')
            p_file = open(p_filename, 'r')
            last = ''
            for r in p_file:
                last = r
            lp = float(last.split(',')[4].rstrip('\n').rstrip('0'))
            ans[p.rstrip('\n')]  =  lp
        return ans

    #Helper function probably doesn't belong here but its here anyway
    def expMean(self, ts,lag=20):
        K = 2./(lag+1)
        #K = 1./float(lag)
        notyet = True
        Values = ts.Values()
        mdates = []
        mvalues = []

        for i in range(0,len(ts.Values())):            
            if notyet:
                mean = Values[i]
                notyet = False
            else:
                mean = mean-(K*(mean-Values[i]))
                mvalues.append( mean )

        return mvalues[-1]

if __name__ == "__main__":
    p = PairConverter('test')
    print p.getDollarsPerPoint('CHFJPY')
    '''print p.getAbsoluteMls('XAUUSD')
    print p.scaleByCorrelation('EURCHF','XAUUSD')
    print p.pairRatio('EURCHF', 'XAUUSD')'''
    '''pair_file = os.path.join(os.environ['CODE_DIR'], 'pairs')
    pc = PairConverter(pair_file)
    all_pairs = []
    pair_f = open(pair_file, 'r')
    for pairs in pair_f:
        all_pairs.append(pairs.rstrip('\n'))
    for p1 in all_pairs:
        for p2 in all_pairs:
            a = pc.pairRatio(p1, p2)'''
