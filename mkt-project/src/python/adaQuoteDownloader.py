#!/bin/python2.5

import time
import datetime
import urllib
import os
import sys
from HTMLParser import HTMLParser

class MyHTMLParser(HTMLParser):
    __open = 0.0
    __high = 0.0
    __low = 0.0
    __close = 0.0
    __next_open = False
    __next_low_high = False
    __next_low_high_count = 0
    __next_close = False

    def handle_data(self, data):
        if self.__next_open:
            self.__open = float(data.replace(',', ''))

        if self.__next_close:
            self.__close = float(data.replace(',', ''))

        if self.__next_low_high:
            self.__next_low_high_count += 1
            if self.__next_low_high_count == 1:
                self.__low = float(data.replace(',', ''))
            if self.__next_low_high_count == 3:
                self.__high = float(data.replace(',', ''))
            if self.__next_low_high_count == 3:
                self.__next_low_high = False


        if data == 'Last Trade:':
            self.__next_close = True
        else:
            self.__next_close = False

        if data == 'Open:':
            self.__next_open = True
        else:
            self.__next_open = False

        if data == 'Day\'s Range:':
            self.__next_low_high = True
            self.__next_low_high_count = 0
            
    def GetOHLC(self):
        return (self.__open, self.__high, self.__low, self.__close)

def download_quote(quote):
    params = urllib.urlencode({'s': quote + '=X' })
    urlname = "http://finance.yahoo.com/q"
    f = urllib.urlopen(urlname, params)
    parser = MyHTMLParser()
    parser.feed(f.read())
    parser.close()
    return parser.GetOHLC()

def process(quote, data_dir = os.environ['DATA_DIR']):
    filename = os.path.join(data_dir, '$' + quote.upper() + '.txt')
    if not os.path.exists(filename):
        print 'file %s for quote %s not found.' %(filename, quote)
        return
    else:
        print 'processing %s for pair %s' %(filename, quote)
    f = open (filename, 'r')
    lines = f.readlines()
    last_date = lines[-1].split(',')[0]
    parts = last_date.split('/')
    year = int(parts[2])
    month = int(parts[1])
    date = int(parts[0])
    previous_date = datetime.date(year, month, date)
    next_date = previous_date + datetime.timedelta(days=1)
    today = datetime.date.today()
    if today == previous_date:
        print 'prices for %s already in. Ignoring.' %today
        return None
    else:
        if not next_date == today:
            print 'Data missing for days between %s and %s. Please make sure it was a weekend.' %(next_date, today)
    

    if today.weekday() == 5 or today.weekday() == 6:
        print '%s is a weekend. Script should not be running.' %today
        return
    print 'Downloading quotes for %s' %(today)
    (o, h, l, c) = download_quote(quote)
    if quote == 'XAUUSD' or quote == 'XAGUSD':
        t = h
        h = l
        l = t
    today_tuple = today.timetuple()
    l = '%d/%d/%d,%f,%f,%f,%f\n' %(today_tuple[2], 
                                   today_tuple[1], 
                                   today_tuple[0], 
                                   o, h, l, c)
    f = open(filename, 'a')
    f.write(l)
    f.close()
    

def main():
    args = sys.argv[1:]
    if not args:
        print 'Usage: %s [quote file]' %(sys.argv[0])
        sys.exit(1)
    
    quote_filename = args[0]
    print 'Extracting pairs from %s' %(quote_filename)
    quote_file = open (quote_filename, 'r')
    for quote in quote_file:
        process(quote[:-1])

if __name__=="__main__":
    main()
