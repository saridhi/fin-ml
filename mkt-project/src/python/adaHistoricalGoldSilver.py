import re
import sys
import string
import os
import urllib
from string import lower
from HTMLParser import HTMLParser

class Table(list):
    pass
	
class Row(list):
    pass

class Cell(object):
    def __init__(self):
        self.data = None
        return
    def append(self,item):
        if self.data != None:
	    print "Overwriting %s"%self.data
        self.data = item

# Get the item on the top of a stack
def top(x):
    return x[len(x)-1]

class TableParser(HTMLParser):
#     _tag = None
#     _buf = ''
#     _attrs = None
#     doc = None # Where the document will be stored
#     _stack = None

    def handle_starttag(self, tag, attrs):
        print tag
#         self._tag = tag
# 	self._attrs = attrs
# 	if lower(tag) == 'table':
# 	    self._buf = ''
#             self._stack.append(Table())
# 	elif lower(tag) == 'tr':
# 	    self._buf = ''
#             self._stack.append(Row())
# 	elif lower(tag) == 'td':
# 	    self._buf = ''
#             self._stack.append(Cell())
	
        #print "Encountered the beginning of a %s tag" % tag

    def handle_endtag(self, tag):
        print tag
# 	if lower(tag) == 'table':
# 	    t = None
# 	    while not isinstance(t, Table):
#                 t = self._stack.pop()
# 	    r = top(self._stack)
#             r.append(t)

# 	elif lower(tag) == 'tr':
# 	    t = None
# 	    while not isinstance(t, Row):
#                 t = self._stack.pop()
# 	    r = top(self._stack)
#             r.append(t)

# 	elif lower(tag) == 'td':
# 	    c = None
# 	    while not isinstance(c, Cell):
#                 c = self._stack.pop()
# 	    t = top(self._stack)
# 	    if isinstance(t, Row):
# 	        # We can not currently have text and other table elements in the same cell. 
# 		# Table elements get precedence
# 	        if c.data == None:
#                     t.append(self._buf)
# 		else:
# 		    t.append(c.data)
# 	    else:
# 	        print "Cell not in a row, rather in a %s"%t
#         self._tag = None
#         #print "Encountered the end of a %s tag" % tag

    def handle_data(self, data):
        print data

if __name__=="__main__":
    urlname = "http://www.kitco.com/londonfix/gold.londonfix08.html"
    f = urllib.urlopen(urlname)
    l = f.read()
    l = l.replace('</SCRIPT\\>', '</SCRIPT>')
    parser = TableParser()
    parser.feed(l)
    parser.close()
