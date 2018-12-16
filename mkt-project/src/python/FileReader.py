# -*- coding: cp1252 -*-
import sys
sys.path.append('Y:\\Nyk\Research\Technical Analysis\Pavitra Kumar\OASIS_Library')
sys.path.append('Y:\\Nyk\Research\Technical Analysis\Pavitra Kumar\OASIS_Library\Sarin')
#from ether import Ether, USECOM
#from fxstdtimeserie import spotCloseTS, spotLowTS, spotHighTS
import datetime, time
import math
from scipy import stats
from numpy import std
import array
#import days
#import startup

#import#win32com.client
#import pythoncom

#from grooming import cachingDecorator, cansetDecorator
#from graphelements import Apple, ConvertDate




class Filereader:
# Wrapper class that includes Technical Analysis functions and Time Series creation function. Also backtests.
# Author: pavitra kumar


    
       
    def createTS(self,dates,values):
        self.dates = dates
        self.values = values
        
                
    def Dates(self):
        return self.dates

    def Values(self):
        return self.values

    def Path(self,asset = 'USD',base = 'JPY'):
        
        return asset+base+".txt"

    def Path2(self):
        return "eurgbpadxvalues.txt"

    def createTStxt(self,asset,base,priceflag='All'):           
            path = self.Path(asset,base)
            f=open(path, 'r')
            dates=[]
            ovalues = []
            hvalues = []
            lvalues = []
            cvalues = []
            while 1:
                line=f.readline()

                #print line
            
                if not line:
                    break
                parts=line.split(',')

                #print parts[2]
                d=parts[0].split('/')
                #print d
                dates.append(datetime.date(int(d[2]), int(d[1]), int(d[0])))
                ovalues.append(float(parts[1]))
                hvalues.append(float(parts[2]))
                lvalues.append(float(parts[3]))
                cvalues.append(float(parts[4]))
                values = [ovalues,hvalues,lvalues,cvalues]
                

            if priceflag == 'O':   
                self.createTS(dates,ovalues)
                return Ether.Creator('TimeSerie', Dates = dates, Values = ovalues)

            elif priceflag == 'H':
                self.createTS(dates,hvalues)
                return Ether.Creator('TimeSerie', Dates = dates, Values = hvalues)
            
            elif priceflag == 'L':
                self.createTS(dates,lvalues)
                return Ether.Creator('TimeSerie', Dates = dates, Values = lvalues)

            elif priceflag == 'C':
                self.createTS(dates,cvalues)
                return Ether.Creator('TimeSerie', Dates = dates, Values = cvalues)

            else:
                self.createTS(dates,values)
                return Ether.Creator('TimeSerie', Dates = dates, Values = ovalues),Ether.Creator('TimeSerie', Dates = dates, Values = hvalues),Ether.Creator('TimeSerie', Dates = dates, Values = lvalues),Ether.Creator('TimeSerie', Dates = dates, Values = cvalues)

    def createTSpickle(self,asset,base):
        
        fO=open(asset+base+"O.txt", 'r')
        fH=open(asset+base+"H.txt", 'r')
        fL=open(asset+base+"L.txt", 'r')
        fC=open(asset+base+"C.txt", 'r')
        fD=open(asset+base+"D.txt", 'r')
        import cPickle as pickle
        ovalues = pickle.load(fO)
        hvalues = pickle.load(fH)
        lvalues = pickle.load(fL)
        cvalues = pickle.load(fC)
        dates = pickle.load(fD)
        values = [ovalues,hvalues,lvalues,cvalues]
        self.createTS(dates,values)
        return Ether.Creator('TimeSerie', Dates = dates, Values = ovalues),Ether.Creator('TimeSerie', Dates = dates, Values = hvalues),Ether.Creator('TimeSerie', Dates = dates, Values = lvalues),Ether.Creator('TimeSerie', Dates = dates, Values = cvalues)


    
    def readFile(self):           
            path2 = self.Path2()
            f=open(path2, 'r')
            dates=[]
            values = []
            while 1:
                line=f.readline()

                #print line
            
                if not line:
                    break
                parts=line.split(',')

                #print parts[2]
                d=parts[0].split('/')
                #print d
                dates.append(datetime.date(int(d[2]), int(d[1]), int(d[0])))
                values.append(float(parts[1]))

            self.createTS(dates,values)
            return Ether.Creator('TimeSerie', Dates = dates, Values = values)

def main():
    
    t = Filereader()
    #print list(t.createTStxt('USD','JPY',priceflag = 'C').Values())
    #print list(t.readFile().Values())
    print (t.createTSpickle('USD','CHF')[0])
    print list((t.createTSpickle('USD','CHF')[0]).Values())
    #print list(t.createTSpickle('USD','CHF')[0].Values())
    #print s.rMean(lag=20)
    #print list(s.rMean(lag=20).Values())
    #print list(s.Values())
    #print list(t.entrySignalsBollinger().Values())
    #print s.calculateReturnsBollinger(t.entrySignalsBollinger())
    
    
    
        
if __name__=="__main__":
	main()
