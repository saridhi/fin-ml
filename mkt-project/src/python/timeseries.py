#import scipy
import datetime


def ConvertDate( value ):
    if type(value) == datetime.date:
        return value
    elif type(value) == datetime.datetime:
        return value.date()
    elif type(value) == pywintypes.TimeType:
        def ole2datetime(oledt):
            return datetime.datetime(1899, 12, 30, 0, 0, 0) + datetime.timedelta(days=float(oledt))
        return ole2datetime(value).date()
    else:
        raise ValueError('Cannot be casted into a date!')

def Convert( value ):
        if (type(value) is tuple) or (type(value) is list):
            return [ Convert(v) for v in value ]
        else:
            try:
                return ConvertDate( value )
            except:
                return value
            
class TimeSeries:

    def __init__(self, dates, values):
        self.dates = dates
        self.values = values

    def Dates(self):
        return self.dates

    def Values(self):
        return self.values

    def HasDate(self, date):
        """ Checks if a given date appears in this time-serie """
        try:
            gdate = ConvertDate(date).date()
        except:
            gdate = ConvertDate(date)
        try:
            return (self.GDates().index(gdate) >= 0)
        except:
            return False
        
    def Changes(self, isFwd = False):
        """ Returns the time-serie of the daily variations """

        if not(isFwd):
            dates = self.Dates()[1:]
        else:
            dates = self.Dates()[0:-1]
        
        rvalues = self.Values()[1:]
        lvalues = self.Values()[0:-1]

        if not (len(rvalues) == len(lvalues)):
            print 'Error! len(rvalues) different from len(lvalues)'
            sys.exit(1)

        changes = []
        size = len(rvalues)
        for i in range(size):
            changes.append(rvalues[i] - lvalues[i])

        if len(changes) >= 1:
            return TimeSeries(dates = dates, values = changes)
        else:
            raise ValueError('Could not calculate any value -the window is probably too long-')

    def Floor(self, floor = 0.):
        """ Returns a time-serie whose values are floored """
        def flooring(x):
            if x>=floor:
                return x
            else:
                return floor
        return TimeSeries(dates = self.Dates(), values = [flooring(x) for x in self.Values()])

    def ApplyOperand(self, otherTS, operand = '__add__'):
    
        if type(otherTS) in [int,float]:
                othervalue = float(otherTS)
                cvalues = [ getattr(float(value),operand)(othervalue) for value in self.Values() ]
                if len(self.Dates()) > 0:
                    return TimeSeries(dates = self.Dates(), values = cvalues)
                else:
                    raise ValueError('Could not calculate any value -not any date in the TS-')
        if len(self.Dates()) == len(otherTS.Dates()):
            Test = True
            for d, dd in zip(Convert(self.Dates()), Convert(otherTS.Dates())):
                if type(d) == datetime.datetime:
                    d = d.date()
                if type(dd) == datetime.datetime:
                    dd = dd.date()
                Test = Test and (d == dd)
            if Test:
                # The two series have the same dates so that is easy
                Values = self.Values()
                otherValues = otherTS.Values()        
                cdates = self.Dates()
                cvalues = []
                for i in range(0,len(self.Dates())):
                    cvalues.append( getattr(float(Values[i]),operand)(float(otherValues[i])) )
                if len(cvalues) > 0:
                    return TimeSeries(dates = cdates, values = cvalues)
                else:
                    raise ValueError('Could not calculate any value -not any date in the TS-')
        # The two series have different dates
        cdates = []
        cvalues = []
        for i, date in zip(range(0,len(self.Dates())), self.Dates()):
            if otherTS.HasDate(date):
                otherval = otherTS.Fetch(date)
                cvalues.append( getattr(float(self.Values()[i]),operand)(float(otherval)) )
                cdates.append( date )
        if len(cvalues) > 0:
            return TimeSeries(dates = cdates, values = cvalues)

    def Add(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__add__')

    def __add__(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__add__')
    

    def Subtract(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__sub__')

    def __sub__(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__sub__')
    

    def Multiply(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__mul__')

    def __mul__(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__mul__')


    def Divide(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__div__')

    def __div__(self, otherTSdis):
        return self.ApplyOperand(otherTSdis, '__div__')

    def Fetch(self, date):
        """ Returns the time-serie value corresponding to a given date """
        try:
            gdate = ConvertDate(date).date()
        except:
            gdate = ConvertDate(date)
        try:
            ind = self.GDates().index( gdate )
        except:
            raise ValueError('Date not found')
        return self.Values()[ ind ]

    def GDates(self):
        try:
            return [ConvertDate(d).date() for d in self.Dates()]
        except:
            return [ConvertDate(d) for d in self.Dates()]
        
    def RExpMean(self, lag = 30):
        mdates = []
        mvalues = []

        Dates = self.Dates()
        Values = self.Values()

        K = 2./(lag+1)
        notyet = True
        
        for i in range(0,len(Dates)):            
            if notyet:
                datelag = Dates[i] - datetime.timedelta(lag)
                if datelag > Dates[0]:                
                    j = 0
                    for ind, d in zip(range(0,i),Dates[0:i]):
                        if d<datelag:
                            j = ind

                    mean = sum(Values[j:i]) / len(Values[j:i])
                    notyet = False
            else:
                mean = mean + (Values[i]-mean)*K
                datelag2 = Dates[i] - datetime.timedelta(lag) - datetime.timedelta(lag)
                if datelag2 > Dates[0]:  
                    mdates.append( Dates[i] )
                    mvalues.append( mean )

        if len(mvalues) >= 1:
            return TimeSeries(dates = mdates, values = mvalues)
        else:
            raise ValueError('Could not calculate any value -the window is probably too long-')
    
