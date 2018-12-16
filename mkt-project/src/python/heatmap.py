import sys
import math
import numpy
import copy

colorArray = ['#FF6600',
'#FF7312',
'#FF8224',
'#FF8C36',
'#FF9948',
'#FFA55A',
'#FFB16C',
'#FFB178',
'#FFB98A',
'#FFC59C',
'#FFD2B0',
'#FFDCC2',
'#FFE6D7',
'#FFF0E9',
'#FFFFFF',
'#F8FFF8',
'#EDFFED',
'#DBFFDB',
'#D4FFD4',
'#C7FFC7',
'#B5FFB5',  
'#A3FFA3',
'#7FFF7F',
'#6DFF6D',
'#5BFF5B',
'#37FF37',
'#24FF24',
'#12FF12',
'#00FF00']

def getColorDico():
  colorRange = -7.0 #Minimum possible heatValue
  colorDico = {}
  for c in colorArray:
    colorDico[str(colorRange)]=c
    colorRange = colorRange+0.5
  return colorDico

#Limit heatvalue to -7 or 7
def getColorDicoForPortal(value):
  if (value < -7):
    value = -7
  if (value > 7):
    value = 7
  colorRange = -7.0 #Minimum possible heatValue
  colorDico = {}
  for c in colorArray:
    colorDico[colorRange]=c
    colorRange = colorRange+0.5
  return colorDico[value]
    
def getColorArray():
  return colorArray

def mapToColorArray(diff,maxheatValue,minheatValue):#if heatvaluerange with multiple indicators > len(colorArray)
  colorDico = {}

  for i in range(0,len(colorArray)+diff-2):
    if i < diff:
        colorDico[str(float(maxheatValue-(i*0.5)))]=colorArray[len(colorArray)-1]
        colorDico[str(float((minheatValue+(i*0.5))))]=colorArray[0]
    else:
        colorDico[str(float((minheatValue+(i*0.5))))]=colorArray[int(i+1-diff)]
  return colorDico
  
def mapToHeatValueRange(diff,maxheatValue,minheatValue):# if len(colorArray) > heatvaluerange with multiple indicators
  colorDico = {}
  colorDico[str(maxheatValue)]=colorArray[len(colorArray)-1]
  colorDico[str(minheatValue)]=colorArray[0]

  for i in range(diff,diff+((4*maxheatValue)-1)):
        colorDico[str(float((minheatValue+((i+1-diff)*0.5))))]=colorArray[i]
  return colorDico
  
def getColorDicoMultipleIndicators(maxheatValue,minheatValue):
  
  colorRange = minheatValue
  colorDico = {}
  
  newHeatValueRange = (4*maxheatValue)+1#length of heat value range
  if newHeatValueRange > len(colorArray):
    diff = (((newHeatValueRange) - (len(colorArray)-2))/2)#diff = number of heatvalues to assign to extremes of colorArray if heatvaluerange > len(colorArray), and vice versa if len(colorArray) > heatvaluerange
    colorDico = mapToColorArray(diff,maxheatValue,minheatValue)
      
  else:
    diff = (((len(colorArray)) - (newHeatValueRange-2))/2)
    colorDico = mapToHeatValueRange(diff,maxheatValue,minheatValue)
  return colorDico
  
#selection sort
def selection_sort ( lista ):
  for i in range ( 0, len ( lista ) ):
    min = i
    for j in range ( i + 1, len ( lista ) ):
      if lista[j][1] < lista[min][1] :
        min = j
    lista[i], lista[min] = lista[min], lista[i]
  return lista

def generateBox(xyList, length = 0, currentIndex = 0):
    if (currentIndex == length):
        return (int((length+1)/2)-1,int(length+1)/2-1)
    xyList[currentIndex]=generateBox(xyList, length, currentIndex+1)
    x,y = xyList[currentIndex]
    if (x>y):
        x,y = xyList[currentIndex+1]
        return (x-1,y)
    elif (y>x):
        return (y,x)
    else:
        return (x-1,y)
    
def generateHM(boxsize = 16):
    finalList = []
    xyList = range(0,((boxsize * 2)-1))
    generateBox(xyList, ((boxsize * 2)-1), 0)
    return xyList

def hMapRows(L):
  hMap = [()]
  for i in range(0,L):#hm is ordered diagonally from most bearish at the beginning to most bullish at the end. Indices are across the rows.
                      #To order from most bullish to most bearish, need to reverse hm ordering.
      for j in range(i*L,L*(i+1)):
          hMap[j] = ((L-1-j)+(L*i),L-i-1)
          hMap.append(hMap)
  return hMap

def maxLen(LenSec):
  boxsize = math.sqrt(LenSec)
  if (boxsize != int(boxsize)):
        maxLen = boxsize+1
  else:
        maxLen = boxsize
  return maxLen
  
def heatMap(securityList = [{'securityName':'EURUSD', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'Yes','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                          {'securityName':'GBPJPY', 'MA1':'Bearish', 'MA2':'Bullish', 'MACrossover':'Bearish', 'ADX':'Ranging','RSI':'Neutral',
                           'Stochastics':'Bullish','MACD':'Neutral',
                           'Doji':'','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                            {'securityName':'USDJPY', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                             {'securityName':'USDCHF', 'MA1':'Bearish', 'MA2':'Bullish', 'MACrossover':'Bearish', 'ADX':'Ranging','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bearish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                             {'securityName':'EURGBP', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Overbought','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'Bullish','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':'Yes'}],securities = []):

    hMap = []
    total = 1
    valueList,maxheatValue,minheatValue = heatValues(securityList,securities)
    LenSec = len(valueList)

    mL = int(maxLen(LenSec))
    square = mL*mL

    #med = median([k[1] for k in valueList])
    colDico = getColorDicoMultipleIndicators(maxheatValue,minheatValue)#NEW COLOR FUNCTION
    #colDico = getColorDico()
    
    while (len(valueList) < square):
      valueList.append(('',0.0)) #dummy value'''


    sortedList = selection_sort(valueList) #Sort list by heatValue
    while (total <= mL):
        hMap = hMap + generateHM(total)
        total = total + 1

    orderedList = reOrder(sortedList,hMap, mL)
    index = 0
    securityArray = []
    indexArray = []
    colorArray = []
    heatValArray = []

    for ol in orderedList:
      secName = ol[0]
      color = colDico[str(ol[1])]
      index = index + 1
      indexArray.append(index)
      securityArray.append(secName)
      colorArray.append(color)
      heatValArray.append(ol[1])
      
    return {'securityName':securityArray, 'Index':indexArray, 'Color':colorArray, 'heatValue':heatValArray}

def median(numbers):
   
   # Sort the list and take the middle element.
   n = len(numbers)
   copy = numbers[:] # So that "numbers" keeps its original order
   copy.sort()
   if n & 1:         # There is an odd number of elements
      return copy[n // 2]
   else:
      return (copy[n // 2 - 1] + copy[n // 2]) / 2

"A sorted list but map this to the x,y coordinates from generateHM"
def reOrder(valueList, hMap, boxsize):
    count = 0
    dicoOL = {}
    xx = 0
    yy = 0
    orderedArray = []

   # square = boxsize * boxsize
    
    #while (len(valueList) < square):
     # valueList.append({'Dummy':100}) #dummy value
    
    for i in valueList:
      dicoOL[hMap[count]]=i
      count = count + 1

    for yy in range(0, boxsize):
      for xx in range(0,boxsize):
          
        try:
          orderedArray.append(dicoOL[(xx,yy)])
        except:
          None
      xx = 0
    return orderedArray

'''retrieve top two drinks based on absolute ranking and a modified drink selection criteria'''
def absoluteRank(securityList = [{'securityName':'XAUUSD', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'Yes','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                          {'securityName':'EURUSD', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'Yes','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''}]):

    hMap = []
    total = 1
    valueList = heatValuesForPortal(securityList)
    '''print valueList
    valueList = [('EURJPY', -1.0), ('CHFJPY', 1.0), ('GBPUSD', -1.0), ('NZDJPY', 1.0), ('USDCHF', -3.0), ('AUDJPY', -2.0), ('USDCAD', -1.0), ('XAGUSD', 3.0), ('EURCHF', 0.5), ('EURUSD', 3.0), ('XAUUSD', -3.0), ('GBPJPY', -1.0), ('GBPCHF', -0.5), ('USDSGD', 3.0), ('USDSEK', 1.5), ('EURNZD', 2.0), ('USDMXN', 0.5), ('CADJPY', 0.0), ('EURCAD', -1.5), ('AUDCAD', 1.5), ('GBPCAD', 3.5), ('AUDUSD', -2.0), ('EURGBP', 1.0), ('AUDNZD', -3.0), ('USDJPY', 0.5), ('NZDCAD', 3.0), ('NZDUSD', 1.0), ('EURAUD', 3.0)]'''
    securityList = [i[0] for i in valueList] 
    scoreList = [i[1] for i in valueList] #List of original scores maintained here

    absValueList = []
    '''Create the same valueList but with Absolute Values'''
    for symbolScoreTuple in valueList:
      absScore = numpy.abs(symbolScoreTuple[1])
      absValueList.append((symbolScoreTuple[0], absScore))
    
    sortedList = selection_sort(absValueList) #Sort list by heatValue 
    sortedList.reverse() #Highest absolute value comes first

    allAbsScores = []  #Array of unique absolute scores
    for s in sortedList:
      try:
        allAbsScores.index(s[1])
      except:
        allAbsScores.append(s[1])

    uniqueScores = {}
    '''Convert to a dictionary keyed by scores with values as securities having the keyed score'''
    for symbolScoreTuple in sortedList:
      try:
        pairForScore = uniqueScores[symbolScoreTuple[1]]
        pairForScore.append(symbolScoreTuple[0])
        uniqueScores[symbolScoreTuple[1]]=pairForScore
      except:
        symbolArray = []
        symbolArray.append(symbolScoreTuple[0])
        uniqueScores[symbolScoreTuple[1]]=symbolArray
    
    uniqueScores = sortArraysByChaudha(uniqueScores)
    conflictFree, found = tryConflictFreeCombos(uniqueScores, allAbsScores)
    if not(found):
      allLongs = convertToLongs(uniqueScores, allAbsScores, scoreList, securityList)
      barChoice = chaudhaOptimiser(allLongs, securityList, allAbsScores, scoreList)
      chosenDrinks = bushOptimiser(allLongs, barChoice, securityList, allAbsScores)
    else:
      chosenDrinks = conflictFree

    symbolWithScore = []
    print 'Chosen Drinks: ', chosenDrinks
    
    for i in chosenDrinks:
        symbol = ''
        rightSymbol, inverted = symbolAvailable(i[:3], i[3:], securityList)
        newSymbol = invertSymbol(i, inverted)
        symbolWithScore.append((newSymbol, scoreList[securityList.index(newSymbol)]))
    return symbolWithScore, valueList

def invertSymbol(i, inverted):
    if inverted:
           symbol = i[3:] + i[:3]
    else:
           symbol = i[:3] + i[3:]
    return symbol

'''basic search for conflict free combos'''
def tryConflictFreeCombos(uniqueSymbols, allAbsScores):
        for number in allAbsScores:
            symbolList = uniqueSymbols[number]
            first = True
            for symbol in symbolList:
              if first:
                a, b = symbol[:3], symbol[3:]
                first = False
                continue
              else:
                c, d = symbol[:3], symbol[3:]
              if a not in (b,c,d) and b not in (a,c,d) and c not in (a,b,d) and d not in (a,b,c):
                #Give priority to conflict free drinks - only need to do this for highest scores
                return [a+b, c+d], True
            break
        return uniqueSymbols, False
     
'''sort symbols by preference of chaudhaRatios'''
def sortArraysByChaudha(uniqueSymbols):
        from addSpreadToSMRecords import Spread
        spread = Spread()
        print 'Getting chaudha ratios'
        spreadRatios = spread.getSpreadRatios()
        for number in uniqueSymbols.keys():
           symbols = uniqueSymbols[number]
           symbolChaudhaRatio = []
           for s in symbols:
               spreadRatio = spreadRatios[s]
               symbolChaudhaRatio.append((s, spreadRatio))
           sortedSymbols = selection_sort(symbolChaudhaRatio)
           uniqueSymbols[number]=[x[0] for x in sortedSymbols]
        return uniqueSymbols

'''check to see if symbol or its inverse is available in oegg'''
def symbolAvailable(asset, base, securities):
        symbol = asset+base
        inverseSymbol = base+asset
        existingSymbol = ''
        inverted = False
        if symbol in (securities):
            existingSymbol = symbol
        #Flip around asset/base to see if the symbol is available
        elif inverseSymbol in (securities):
            existingSymbol = inverseSymbol
            inverted = True
        if existingSymbol != '':
           return True, inverted
        else:
           return False, inverted

def convertToLongs(uniqueScores, allAbsScores, rawScores, securityList):
       allLongs = {}
       for number in allAbsScores:
           longSymbols = []
           symbolList = uniqueScores[number]
           for symbol in symbolList:
               asset, base = symbol[:3], symbol[3:]
               index = securityList.index(symbol)
               rawScore = rawScores[index]
               if rawScore < 0:
                   newBase = asset
                   newAsset = base
                   longSymbols.append(newAsset+newBase)
               else:
                   longSymbols.append(symbol)
           allLongs[number]=longSymbols
       return allLongs
                 
def chaudhaOptimiser(uniqueScores, securityList, allAbsScores, scoreList):
       securityList = copy.copy(securityList)
       uniqueScores = copy.copy(uniqueScores)
       barChoice = uniqueScores
       for number in allAbsScores:
           newSymbols = []
           filteredSymbols = []
           symbolList = barChoice[number]
           for symbol in symbolList:
               filteredSymbols.append(symbol)
               if len(filteredSymbols)==1: #first symbol
                   continue
               else:
                   newSymbol, index = cancelCommonParts(filteredSymbols, securityList)
                   if newSymbol != '': #Successful cancellation
                       filteredSymbols.pop(index)
                       filteredSymbols.pop(-1)
                       filteredSymbols.append(newSymbol)
                       newSymbols.append((filteredSymbols.index(newSymbol), newSymbol))
           symbolScoreTuple = []
           for i,s in newSymbols:
               try:
                 symbolScoreTuple.append((s,scoreList[securityList.index(s)]))
               except:
                 symbolScoreTuple.append((s,scoreList[securityList.index(invertSymbol(s,True))]))
           sortedList = selection_sort(symbolScoreTuple)
           sortedList.reverse()
           for i,s in zip(sortedList, newSymbols):
               filteredSymbols.pop(s[0])
               filteredSymbols.insert(s[0],i[0])
           barChoice[number]=filteredSymbols
       return barChoice

def cancelCommonParts(symbols, allSecurities):
       asset1, base1 = symbols[-1][:3], symbols[-1][3:]
       newSymbol = ''
       exists = False
       for symbol in symbols:
           asset2, base2 = symbol[:3], symbol[3:]
           if (asset1 == base2):
               exists, inverted = symbolAvailable(asset2, base1, allSecurities)
               newSymbol = asset2+base1
           elif (base1 == asset2):
               exists, inverted = symbolAvailable(asset1, base2, allSecurities)
               newSymbol = asset1+base2
           if exists==True:
               break
           else:
               newSymbol = ''
       return newSymbol, symbols.index(symbol)

def bushOptimiser(uniqueScores, barChoice, symbolList, allAbsScores):
       lastPopped = ''
       chosenTwo = []
       for number in allAbsScores:
           compromisePop = False
           '''chaudhaArray = barChoice[number]
           extendedChoice = removeBarChoice(uniqueScores[number], chaudhaArray)'''
           symbolList = barChoice[number]
           '''+extendedChoice'''
           for symbol in symbolList:
               chosenTwo.append(symbol)
               if len(chosenTwo)==1: #first symbol
                   continue
               else:
                   asset1, base1 = chosenTwo[0][:3], chosenTwo[0][3:]
                   asset2, base2 = chosenTwo[1][:3], chosenTwo[1][3:]
                   if (asset1 == asset2) or (base1 == base2):
                       lastPopped = chosenTwo.pop(-1)
                   elif (asset1 == base2) or (base1 == asset2):
                       lastPopped = chosenTwo.pop(-1)
               if len(chosenTwo) == 2:
                   break
           if (len(chosenTwo)==1) and (lastPopped!=''):
               chosenTwo.append(lastPopped)
               theTwo = copy.copy(chosenTwo)
               if number!=max(allAbsScores):
                   compromisePop = True
           if (len(chosenTwo) == 2):
               break
       
       theTwo = copy.copy(chosenTwo)
       if compromisePop:
          #Untangle the higher score symbols ie reverse the chaudhaOptimiser
          topListSingled = copy.copy(uniqueScores)
          topList = topListSingled[max(allAbsScores)]
          for symbol in topList:
            topListSingled[max(allAbsScores)] = [symbol]
            chosenTwo, compromisePop = bushOptimiserAlternative(uniqueScores, topListSingled, symbolList, allAbsScores)
            if not(compromisePop):
              break
            else:
              chosenTwo = theTwo
       return chosenTwo

def bushOptimiserAlternative(uniqueScores, barChoice, symbolList, allAbsScores):
       lastPopped = ''
       chosenTwo = []
       for number in allAbsScores:
           popped = False
           symbolList = barChoice[number]
           for symbol in symbolList:
               chosenTwo.append(symbol)
               if len(chosenTwo)==1: #first symbol
                   continue
               else:
                   asset1, base1 = chosenTwo[0][:3], chosenTwo[0][3:]
                   asset2, base2 = chosenTwo[1][:3], chosenTwo[1][3:]
                   if (asset1 == asset2) or (base1 == base2):
                       lastPopped = chosenTwo.pop(-1)
                   elif (asset1 == base2) or (base1 == asset2):
                       lastPopped = chosenTwo.pop(-1)
               if len(chosenTwo) == 2:
                   break
           if (len(chosenTwo)==1) and (lastPopped!=''):
               chosenTwo.append(lastPopped)
               popped = True
           if (len(chosenTwo) == 2):
               break
       return chosenTwo, popped

def removeBarChoice(symbolList, toRemove):
       newArray = []
       for i in symbolList:
         try:
           toRemove.index(i)
         except:
           newArray.append(i)
       return newArray
       

'''heatMapRank can only be used specifically for the Portal at the moment'''
def heatMapRank(entries=4, securityList = [{'securityName':'EUR/USD', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'Yes','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                          {'securityName':'GBP/JPY', 'MA1':'Bearish', 'MA2':'Bullish', 'MACrossover':'Bearish', 'ADX':'Ranging','RSI':'Neutral',
                           'Stochastics':'Bullish','MACD':'Neutral',
                           'Doji':'','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                            {'securityName':'USD/JPY', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                             {'securityName':'USD/CHF', 'MA1':'Bearish', 'MA2':'Bullish', 'MACrossover':'Bearish', 'ADX':'Ranging','RSI':'Oversold',
                           'Stochastics':'Oversold','MACD':'Bearish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':''},
                             {'securityName':'EUR/GBP', 'MA1':'Bullish', 'MA2':'Bullish', 'MACrossover':'Bullish', 'ADX':'Trending','RSI':'Oversold',
                           'Stochastics':'Overbought','MACD':'Bullish',
                           'Doji':'Bullish','Engulfing':'','KRD':'','Hammer/SS':'Bullish','DojiActive':'No','EngulfingActive':'',
                           'KRDActive':'','Hammer/SSActive':'Yes'}]):

    hMap = []
    total = 1
    valueList = heatValuesForPortal(securityList)

    sortedList = selection_sort(valueList) #Sort list by heatValue
    topranked, bottomranked = toprank(sortedList),bottomrank(sortedList)
    return topranked, bottomranked


#Most bullish ranking
def toprank(sortedList, entries = 4):
    counter = len(sortedList)
    topentries = []
    
    while counter > 0:
      if (sortedList[counter-1][1])<=0:
       topentries.append(('', '0.0'))
      else:
        topentries.append((sortedList[counter-1][0], sortedList[counter-1][1]))
      counter = counter-1
      if (len(topentries) == (entries)):
        break
    return topentries

#Most bearish ranking
def bottomrank(sortedList, entries = 4):
    counter = 1
    bottomentries = []
    
    while (counter)<= len(sortedList):
      if (sortedList[counter-1][1])>=0:
        bottomentries.append(('', '0.0'))
      else:
        bottomentries.append((sortedList[counter-1][0], sortedList[counter-1][1]))
      if (len(bottomentries) == (entries)):
        break
      counter = counter+1
    return bottomentries

def createTableHeatMap(securities,securityList,heatMap,minBox=4):
        rowcolor = []
        bgcolors = []
        table = []
        row = []
        hMap = [()]

        LenSec = len(securities)
        L = int(maxLen(LenSec))#call maxLen from heatmap
        valueList,maxheatValue,minheatValue = heatValues(securityList,securities)
        
        sec = heatMap['securityName']
        val = heatMap['heatValue']
        
        #call hMapRows from heatmap. The function 'reOrder according to hMapRows' performs reverse ordering on hm, to order diagonally from most bullish to bearish, across the rows.
        orderedsecList = reOrder(sec,hMapRows(L), L)
        orderedvalList = reOrder(val,hMapRows(L), L)
        
        if LenSec < minBox:#print all in a row, excluding blank columns
            for a,b in zip(orderedsecList,orderedvalList):
                if a!='':
                    row.append(a)
                    rowcolor.append(getColorDicoMultipleIndicators(maxheatValue,minheatValue)[str(b)])
            bgcolors.append(rowcolor)
            table.append(row)

        else:#print in a box
            for i in range(0,L):
                
                for a,b in zip(orderedsecList[i*L:L*(i+1)],orderedvalList[i*L:L*(i+1)]):
                    
                    row.append(a)
                    rowcolor.append(getColorDicoMultipleIndicators(maxheatValue,minheatValue)[str(b)])
                    
                bgcolors.append(rowcolor)
                table.append(row)
                row = []
                rowcolor = []
                
        return table, bgcolors

def processHeatmap(securityList,securities,minBox=3):
        heatMapArray = heatMap(securityList,securities)
        table, bgcolors = createTableHeatMap(securities,securityList,heatMapArray,minBox)
        return table, bgcolors

def heatValuesForPortal(sortedList = None):
    returnArray = []
    for s in sortedList:
        secName = s['securityName']
        ma1 = s['MA1']
        ma2 = s['MA2']
        maCross = s['MACrossover']
        adx = s['ADX']
        rsi = s['RSI']
        stochs = s['Stochastics']
        macd = s['MACD']
        doji = s['Doji']
        dojiActive = s['DojiActive']
        engulf = s['Engulfing']
        engulfActive = s['EngulfingActive']
        krd = s['KRD']
        krdActive = s['KRDActive']
        hammer = s['Hammer/SS']
        hammerActive = s['Hammer/SSActive']
        heat = 0.0

        ma1Flag = 0.0
        ma2Flag = 0.0
        maCrossFlag = 0.0
        adxFlag = 0.0
        macdFlag = 0.0
        rsiFlag = 0.0
        stochFlag = 0.0
        dojiFlag = 0.0
        krdFlag = 0.0
        engulfFlag = 0.0
        hammerFlag = 0.0

        if (adx == ''):
          adx = 'Trending'
          
        if adx == 'Trending':
          adxFlag = 1
          if (ma1 == 'Bullish'):
            ma1Flag = 1
          elif (ma1 == 'Bearish'):
            ma1Flag = -1
          if (ma2 == 'Bullish'):
            ma2Flag = 1
          elif (ma2 == 'Bearish'):
            ma2Flag = -1
          if (maCross == 'Bullish'):
            maCrossFlag = 1
          elif (maCross == 'Bearish'):
            maCrossFlag = -1

        if rsi == 'Oversold':
          rsiFlag = 1
        elif rsi == 'Overbought':
          rsiFlag = -1

        if macd == 'Bullish':
          macdFlag = 1
        elif macd == 'Bearish':
          macdFlag = -1

        if stochs == 'Oversold':
          stochFlag = 1
        elif stochs == 'Overbought':
          stochFlag = -1
        elif stochs == 'Bullish':
          stochFlag = 0.5
        elif stochs == 'Bearish':
          stochFlag = -0.5

        if (dojiActive=='Yes'):
          if (doji == 'Bullish'):
            dojiFlag = 1
            if (ma1Flag == -1):
              ma1Flag = 0
            if (ma2Flag == -1):
              ma2Flag = 0
            if (maCrossFlag == -1):
              maCrossFlag = 0
          elif (doji == 'Bearish'):
            dojiFlag = -1
            if (ma1Flag == 1):
              ma1Flag = 0
            if (ma2Flag == 1):
              ma2Flag = 0
            if (maCrossFlag == 1):
              maCrossFlag = 0

        if (engulfActive=='Yes'):
          if (engulf== 'Bullish'):
            engulfFlag = 1
            if (ma1Flag == -1):
              ma1Flag = 0
            if (ma2Flag == -1):
              ma2Flag = 0
            if (maCrossFlag == -1):
              maCrossFlag = 0
          elif (engulf == 'Bearish'):
            engulfFlag = -1
            if (ma1Flag == 1):
              ma1Flag = 0
            if (ma2Flag == 1):
              ma2Flag = 0
            if (maCrossFlag == 1):
              maCrossFlag = 0

        if (hammerActive=='Yes'):
          if (hammer == 'Bullish'):
            hammerFlag = 1
            if (ma1Flag == -1):
              ma1Flag = 0
            if (ma2Flag == -1):
              ma2Flag = 0
            if (maCrossFlag == -1):
              maCrossFlag = 0
          elif (hammer == 'Bearish'):
            hammerFlag = -1
            if (ma1Flag == 1):
              ma1Flag = 0
            if (ma2Flag == 1):
              ma2Flag = 0
            if (maCrossFlag == 1):
              maCrossFlag = 0

        if (krdActive=='Yes'):
          if (krd == 'Bullish'):
            krdFlag = 1
            if (ma1Flag == -1):
              ma1Flag = 0
            if (ma2Flag == -1):
              ma2Flag = 0
            if (maCrossFlag == -1):
              maCrossFlag = 0
          elif (krd == 'Bearish'):
            krdFlag = -1
            if (ma1Flag == 1):
              ma1Flag = 0
            if (ma2Flag == 1):
              ma2Flag = 0
            if (maCrossFlag == 1):
              maCrossFlag = 0
        heat = ma1Flag + ma2Flag +maCrossFlag +rsiFlag +stochFlag +macdFlag+dojiFlag+krdFlag+engulfFlag+hammerFlag
        returnArray.append((secName,heat))    
    return returnArray

if __name__ == "__main__" :
  print absoluteRank()



























  
