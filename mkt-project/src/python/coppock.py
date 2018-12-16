from techseries import maxHist, acTechSeries
import datetime, time
from ether import Ether
from tsmanager import TSManager

class Coppock():
    #calculate a weighted moving average
    def weightedMovingAverage(self,ts,lag=10):

        mdates = []
        mvalues = []
        
        Dates = ts.Dates()
        Values = ts.Values()

        #print Dates,'dates'
        #print Values,'values'
        
        for i in (range(0,len(Dates))):
                print i
                if ((i+lag)>len(Dates)):
                        break
                dt, val = self.auxMeanWeighted(ts, i, (i+lag-1))
                mdates.append( dt )
                mvalues.append( val )
                
        print mdates,'mdates'
        print mvalues,'mvalues'
                        
        return Ether.Creator('TimeSerie', Dates = mdates, Values = mvalues)

    def auxMeanWeighted(self,ts, start, end):
        length = end-start+1
        dayrange = range(1,length+1)
        sumdays = sum(dayrange)
        weightedValues = []
        for i in range(start,end+1):
            #weight each value
            val = (ts.Values()[i])*((float(i+1-start))/(sumdays))
            #print val,'val'
            weightedValues.append(val)
                
        if (length > 0):
            return ts.Dates()[end],(sum(weightedValues[0:length]))

    #calculate Coppock indicator, based on 14 month and 11 month ROC and 10 month weighted moving average
    def coppockIndicator(self,t,lag1=14,lag2=11,lag3=10):
        ROCSum = (t.rROC(lag1)) + (t.rROC(lag2))
        coppock = self.weightedMovingAverage(ROCSum,lag3)
        #coppock.ToExcel("coppock.xls")
        return coppock

def main():    
        print 'start', datetime.datetime.today()
        c = Coppock()
        t=acTechSeries('DJIA','',maxHist, 'M','True')
        print list(t.securityC().Values()),'SECURITY C'
        coppock=c.coppockIndicator(t)
        



if __name__=="__main__":
	main()

                   
