
dicoRGB = {}
dicoRGB['LIGHTORANGE'] = "#FFC59C"
dicoRGB['ORANGE'] = "#F9B40D"
dicoRGB['RED'] = "0xFFB6B6"
dicoRGB['DARKRED'] = "#FF6666"
dicoRGB['LIGHTBLUE'] = "#8CD9C9"
dicoRGB['BLUE'] = "#BBBBFF"
dicoRGB['DARKBLUE'] = "#6666FF"
dicoRGB['WHITE'] = "#FFFFFF"
dicoRGB['BLACK'] = "#000000"
dicoRGB['GREY'] = "#D3D3D3"
dicoRGB['GREEN'] = "0xB6FFB6"
dicoRGB['STATSGREEN'] = "#006600"
dicoRGB['STATSRED'] = "#CC0000"
dicoRGB['Bullish'] = dicoRGB['GREEN']
dicoRGB['Bearish'] = dicoRGB['RED']
dicoRGB['Neutral'] = dicoRGB['WHITE']
dicoRGB['Trending'] = dicoRGB['BLUE']
dicoRGB['Ranging'] = dicoRGB['WHITE']
dicoRGB['Overbought'] = dicoRGB['RED']
dicoRGB['Oversold'] = dicoRGB['GREEN']

from lxml import etree as ET
import trendsXMLParser
import datetime, string, time
from core import days
from mappings import All_Asset_Dates

#This contains functions to generate XML files ready for FLEX to read

#Used for FX Portal
def makePortalXML(title='This is a title', footer='This is a footer', keys=['First Col', 'Second Col'],
            table=[[0.,0.],[0.,0.]], bgcolors=None, helpmsg=None, helpimg=None, tooltip=None):
    tXMLp = trendsXMLParser.trendsXMLParser()
    root = tXMLp.createPortalHeatmap()
    root.find(".//Title").text = title
    root.find(".//Footer").text = footer
    if tooltip != None:
        root.find(".//ToolTip").text = tooltip[0]
        root.find(".//DClickURL").text = tooltip[1]

    '''singularBase = ''
    #Check to see if it is a singular base
    for rowdata in table:
        singularBase = '/'+rowdata[0][-3:]
        for i in range(len(rowdata)):
            if str(rowdata[i][-4:]) == singularBase:
                continue
            else:
                singularBase = ''
                break'''
            
    root.find(".//HelpText").text = helpmsg
    for i in range(len(keys)):
        column = ET.SubElement(root.find(".//Columns"), "Column")
        column.set("tagName", "Col"+str(i))
        displayname = ET.SubElement(column, "DisplayName")
        displayname.text = keys[i]
        
    if bgcolors == None:
        bgcolors = []
        for rowdata in table:
            rowcolor = ['WHITE' for data in rowdata]
            bgcolors.append(rowcolor)

    for rowdata, rowcolor in zip(table, bgcolors):
        row = ET.SubElement(root.find(".//Data"), "Row")
        for i in range(len(rowdata)):
            col = ET.SubElement(row, "Col"+str(i))
            if ((not(rowcolor[i] in ['','WHITE'])) and (str(rowdata[i])!='')):
                if rowcolor[i][0] != '#':
                    col.set("bgRGBColour", dicoRGB[rowcolor[i]])
                else:
                    col.set("bgRGBColour", rowcolor[i])
            col.text = str(rowdata[i])

    return ET.tostring(root, pretty_print =True)

#Heatmap for TRENDS
def makeHeatmapXML(root_element, table=[[0.,0.],[0.,0.]], bgcolors=None):
    trendsXML = trendsXMLParser.trendsXMLParser()
    root = trendsXML.createHeatmap(root_element)
    heatmaproot = root.find(".//Heatmap")

    try:
        cols = len(table[0])
    except:
        cols = 0
    
    for i in range(0,cols):
        column = ET.SubElement(heatmaproot.find(".//Columns"), "Column")
        column.set("tagName", "Col"+str(i))
        displayname = ET.SubElement(column, "DisplayName")
                
    if bgcolors == None:
        bgcolors = []
        for rowdata in table:
            rowcolor = ['WHITE' for data in rowdata]
            bgcolors.append(rowcolor)
     
    for rowdata, rowcolor in zip(table, bgcolors):
        row = ET.SubElement(heatmaproot.find(".//Data"), "Row")
        
        for i in range(len(rowdata)):
            col = ET.SubElement(row, "Col"+str(i))
            if rowcolor[i][0] != '#':
                col.set("bgRGBColour", dicoRGB[rowcolor[i]])
            else:
                col.set("bgRGBColour", rowcolor[i])
            col.text = str(rowdata[i])

    return root

#Legend for heatmap; colors are contained in heatmap.py            
def makeHeatmapLegendXML(root_element, colDico):
    #for heatmaplegend
    trendsXML = trendsXMLParser.trendsXMLParser()
    root = trendsXML.createHeatmapLegend(root_element)
    heatmaplegendroot = root.find(".//HeatmapLegend")

    column0 = ET.SubElement(heatmaplegendroot.find(".//Columns"), "Column")
    column0.set("tagName", "Col0")
    displayname0 = ET.SubElement(column0, "DisplayName")

    from heatmap import selection_sort
    #order heat values from lowest to highest and obtain corresponding colors
    colorValuesKeys = [(y,float(x)) for x,y in zip(colDico.keys(), colDico.values())]
    colorArrayOrdered = selection_sort(colorValuesKeys)
    colorArray = [x for (x,y) in colorArrayOrdered]
        
    rowlegend = ET.SubElement(heatmaplegendroot.find(".//Data"), "Row")
    colzero = ET.SubElement(rowlegend, "Col0")

    if len(colDico) <= 1:
        colzero.text = "Neutral"
        '''colcolor = ET.SubElement(rowlegend, "Col1")
        colcolor.set("bgRGBColour", dicoRGB['Neutral'])
        columnfinal = ET.SubElement(heatmaplegendroot.find(".//Columns"), "Column")
        columnfinal.set("tagName", "Col1")
        displaynamefinal = ET.SubElement(columnfinal, "DisplayName")'''
        return ET.tostring(root, pretty_print =True)
    else:
        colzero.text = "Bullish"
        #obtain list of all unique colors, ordered from most bullish to bearish
        counter = 0
        for i in range(0,len(colorArray)):
            if colorArray[len(colorArray)-1-i]!=colorArray[len(colorArray)-2-i]:
                counter = counter + 1
                col = ET.SubElement(rowlegend, "Col"+str(counter))
                col.set("bgRGBColour",colorArray[len(colorArray)-1-i])
        colend = ET.SubElement(rowlegend, "Col"+str(counter+1))
        colend.text = "Bearish"

        for i in range(0,counter):
            column = ET.SubElement(heatmaplegendroot.find(".//Columns"), "Column")
            column.set("tagName", "Col"+str(i+1))
            displayname = ET.SubElement(column, "DisplayName")

        columnfinal = ET.SubElement(heatmaplegendroot.find(".//Columns"), "Column")
        columnfinal.set("tagName", "Col"+str(counter+1))
        displaynamefinal = ET.SubElement(columnfinal, "DisplayName")

    return ET.tostring(root, pretty_print =True)

#Create snapshot with ordered columns
def makeSnapshotXML(snapShotRows=None,securities = ['EUR/GBP'],indicatorKeys=[]):

    trendsXML = trendsXMLParser.trendsXMLParser()
    root = trendsXML.createSnapshot()
    
    columnSec = ET.SubElement(root.find(".//Columns"), "Column")
    columnSec.set("tagName",'Col0')
    displayname = ET.SubElement(columnSec, "DisplayName")
    displayname.text = 'Security'

    #adxIndicator will be used for Heatmap Legend
    #print indicatorKeys,'indicatorKeys'
    for i in range(0,len(indicatorKeys)):
                column = ET.SubElement(root.find(".//Columns"), "Column")
                column.set("tagName", "Col"+str(i+1))
                displayname = ET.SubElement(column, "DisplayName")
                
                indicatorName = indicatorKeys[i][0].title()
                ID =indicatorKeys[i][1]

                if ID > 1:
                    displayname.text = str(indicatorName) + " " + str(ID)
                else:
                    displayname.text = str(indicatorName)
                    
    for sec in securities:
        
            row = ET.SubElement(root.find(".//Data"), "Row")
            colSec = ET.SubElement(row, "Col"+str(0))
            colSec.text = str(sec)

            #use new set of indicator arrays for each security
            indicatorRow = snapShotRows[securities.index(sec)]

            columnNumber = 0
            for indicators, indications in zip(indicatorKeys, indicatorRow):
                        indicatorName = indicators[0]
                        #print indicatorName,'indicatorName'
                        indicatorId = indicators[1]
                        columnNumber += 1
                        indicatorStatus = indications[0]
                        quickStatCalc = indications[1]

                        col = ET.SubElement(row, "Col"+str(columnNumber))
                        col.text = str(indicatorStatus[1])

                        if (indicatorStatus[1] != ''):
                            if (indicatorStatus[1] != 'Neutral'):
                                    col.set("bgRGBColour", dicoRGB[str(indicatorStatus[1])])
                            if indicatorName in ("doji", "key reversal", "hammer/shooting star", "engulfing"):
                                if str(indicatorStatus[2]) != 'Yes':
                                    col.set("bgRGBColour", dicoRGB['GREY'])
                                    
                        quickStatsPrint = ''
                        if indicatorName in ("doji", "key reversal", "hammer/shooting star", "engulfing", "moving average", "ma crossover", "rsi", "macd", "adx", "slow stochastics"):
                            if not((indicatorName == "slow stochastics" and (indicatorStatus[1].lower() in ('bullish', 'bearish'))) or indicatorStatus[1].lower() in ('neutral')):
                                    bLabel = indicatorStatus[1].lower()
                                    if bLabel == 'overbought':
                                        bLabel = 'bearish'
                                    elif bLabel == 'oversold':
                                        bLabel = 'bullish'
                                    quickStatsString = []
                                    if type(quickStatCalc) == dict:
                                        for quickStatKey in quickStatCalc.keys():
                                            newlabels = ''
                                            newvalues = ''
                                            if quickStatKey == 'Signal date':
                                                newlabels = 'Signal date'
                                                newvalues = str(convertDateFormat(quickStatCalc[quickStatKey]))#+ " (" + str(quickstats['DaysAgo']) + " days ago)"
                                                quickStatsString.insert(0,(newlabels, newvalues))
                                            elif quickStatKey == 'Success':
                                                newlabels = 'Signal odds'
                                                try:
                                                    newvalues = str(float(round((quickStatCalc[quickStatKey]) * 100, 2))) + '% of ' + str(quickStatCalc['Length'])
                                                except:
                                                    newvalues = 'Unknown'
                                                quickStatsString.insert(1,(newlabels, newvalues))
                                            elif quickStatKey == 'ExtremeVal' and indicatorName!="adx":
                                                newlabels = 'Max '+bLabel+' move (recent signal)'
                                                newvalues = str(float(round((quickStatCalc[quickStatKey]) * 100, 2)))
                                                newvalues += "%"
                                                quickStatsString.insert(2,(newlabels, newvalues))
                                            elif quickStatKey == 'CTMoves' and indicatorName!="adx":
                                                newlabels = 'Max '+bLabel+' move (historic average)'
                                                try:
                                                    newvalues = str(float(round((quickStatCalc[quickStatKey]) * 100, 2)))
                                                except:
                                                    newvalues = 'Unknown'
                                                newvalues += "%"
                                                quickStatsString.insert(3,(newlabels, newvalues))
                                
                                    for i in range(0, len(quickStatsString)):
                                        if str(quickStatsString[i][0]) != '':
                                            quickStatsPrint += "<p>" + str(quickStatsString[i][0])+ ": " + "<b>" + str(quickStatsString[i][1]) + "</b>" + "</p>"
                                            
                                    if quickStatsPrint != '':
                                        quickStatsPrint += "<p></p><p>Data since " + All_Asset_Dates[sec].strftime('%B-%Y' + "</p>")
                                    
                                    if str(indicatorStatus[2]) != 'Yes' and indicatorName in ("doji", "key reversal", "hammer/shooting star", "engulfing"):
                                        quickStatsPrint = "<p>Signal date" + ": <b>" + str(convertDateFormat(quickStatCalc['Signal date'])) + "</b></p><p>Pattern voided on" + ": <b>" + str(convertDateFormat(indicatorStatus[3])) + "</b></p>"
                            elif quickStatCalc != '':
                                quickStatsPrint += '<p>Signal date' + ": <b>" + str(convertDateFormat(quickStatCalc)) + "</b>" 

                            if quickStatsPrint!= '':
                                col.set("quickstats", str(quickStatsPrint)) 
    return root

#Make backtest return XML
def makeBacktestXML(returnBacktestStats = [{}],period = 'D'):
    trendsXML = trendsXMLParser.trendsXMLParser()
    root = trendsXML.createBacktest()
    datapointsAvgReturns = ''
    probabilityPoints = ''
    riskAdjustedReturns = ''
    maxDrawDowns = ''
    lastDay = ''

    backtestavgdaily = returnBacktestStats['BacktestAvgDailyReturns']
    backtestsignals = returnBacktestStats['BacktestSignalsReturns']

    if period == 'D':
        periodly = 'daily'
        period = 'day'
    elif period == 'W':
        periodly = 'weekly'
        period = 'week'
    else:
        periodly = 'monthly'
        period = 'month'

    #print backtestsignals,'backtestsignals'
    #print backtestavgdaily,'backtestavgdaily'

    if backtestavgdaily != {}:
        lastDay = str(backtestavgdaily['Dates'][-1])
        #get list of dates and returns for all horizons
        datapointsAvgReturns = ["0,0;"] + [str(x) + "," + str(float(round(y,5)))+ ";" for x,y in zip((backtestavgdaily['Dates']),(backtestavgdaily['Returns']))]
        probabilityPoints = ["0,0;"] + [str(x) + "," + str(float(round(y,2)))+ ";" for x,y in zip((backtestavgdaily['Dates']),(backtestavgdaily['Probabilities']))]
        riskAdjustedReturns = ["0,0;"] + [str(x) + "," + str(float(round(y,5)))+ ";" for x,y in zip((backtestavgdaily['Dates']),(backtestavgdaily['RiskAdjReturns']))]
        maxDrawDowns = ["0,0;"] + [str(x) + "," + str(float(round(y,5)))+ ";" for x,y in zip((backtestavgdaily['Dates']),(backtestavgdaily['MaxDrawDown']))]
        #print 'datapointsAvgReturns',datapointsAvgReturns

    #get list of dates and returns for all signals
    datapointsSignals = ["" + str(convertDateFormat(x)) + "," + str(float(round(y,2)))+ ";" for x,y in zip((backtestsignals['Dates']),(backtestsignals['Returns']))]
    #print 'datapointsSignals',datapointsSignals

    for element in root.iter("ReturnsChart"):
        element.find(".//chart").set("title","")
        element.find(".//chart").set("rightVerticalAxisTitle",'RYAxis')
        element.find(".//chart").set("horizontalAxisTitle",'Dates')
        element.find(".//chart").set("leftVerticalAxisTitle",'% Move')
        element.find(".//dataseperator").text = ";"
        element.find(".//renderer").text = "column,line"
        element.find(".//legend").text = "Returns per signal" 
        element.find(".//visible").text = ''
        element.find(".//axis").text = "left"
        element.find(".//datapoints").text = convertToSingleString(datapointsSignals)
        
    for element in root.iter("AverageReturnsChart"):
        element.find(".//chart").set("title","")
        element.find(".//chart").set("rightVerticalAxisTitle",'RYAxis')
        element.find(".//chart").set("horizontalAxisTitle",str(period))
        element.find(".//chart").set("leftVerticalAxisTitle",'Average '+periodly+' return %')
        element.find(".//legend").text = "Average "+periodly+" return"    
        element.find(".//dataseperator").text = ";"
        element.find(".//renderer").text = "line"
        element.find(".//visible").text = ''
        element.find(".//axis").text = "left"
        element.find(".//datapoints").text = convertToSingleString(datapointsAvgReturns)
        element.find(".//probabilitypoints").text = convertToSingleString(probabilityPoints)
        element.find(".//riskadjustedreturns").text = convertToSingleString(riskAdjustedReturns)
        element.find(".//maxdrawdowns").text = convertToSingleString(maxDrawDowns)

    #print backtestsignals, ' BACKTEST SIGNALS
    
    if backtestsignals['Dates']==[]:
        root.find(".//Statistics").text = "No signals found"
        
    else:
        root.find(".//Statistics").text = "Performance on exit "+period+" "+lastDay+":<br></br>"
        root.find(".//Statistics").text += "<ul><li>Number of signals: <b>" + str(colorStatistics(backtestsignals['NumberOfSignals'], False)) + "</font></b></li>"
        root.find(".//Statistics").text += "<li>Probability of Positive Return: <b>" + str(colorStatistics(backtestsignals['ProbabilityPositiveReturn']))+ "%</font></b></li>"
        root.find(".//Statistics").text +=  "<li>Mean Return: <b>" + str(colorStatistics(backtestsignals['MeanReturnoverSignals'])) + "%</font></b></li>"
        root.find(".//Statistics").text += "<li>Total Return: <b>" + str(colorStatistics(backtestsignals['SumReturnoverSignals'])) + "%</font></b></li>"
        root.find(".//Statistics").text += "<li>Median Return: <b>" + str(colorStatistics(backtestsignals['MedianReturnoverSignals'])) + "%</font></b></li>"
        root.find(".//Statistics").text += "<li>Max Return: <b>" + str(colorStatistics(backtestsignals['MaxReturnoverSignals'])) + "%</font></b></li>"
        root.find(".//Statistics").text += "<li>Min Return: <b>" + str(colorStatistics(backtestsignals['MinReturnoverSignals'])) + "%</font></b></li>"
        root.find(".//Statistics").text += "<li>Standard Deviation of Returns: <b>" + str(colorStatistics(backtestsignals['StDevReturns']))+ "</font></b></li>"
        #root.find(".//Statistics").text += "<li>Initial Investment: <b>" + str(colorStatistics(backtestsignals['Investment'],False))+"</font></b></li>"
        #root.find(".//Statistics").text += "<li>Total Return: <b>" + str(colorStatistics(backtestsignals['EndReturn'])) + "%</font></b></li>"
        print backtestavgdaily['MaxDrawDown'], 'max drawdown'
        root.find(".//Statistics").text += ('<li>Max Drawdown (%s 0-%s): <b>'%(period, lastDay))+ str(colorStatistics(backtestavgdaily['MaxDrawDown'][-1], forceRed=True)) + "%</font></b></li></ul>"
        if len(backtestsignals['Dates'])==1:
            root.find(".//Statistics").text += "<li>Sharpe Ratio: <b>N/A</b></p>"
        else:
            root.find(".//Statistics").text += "<li>Sharpe Ratio: <b>" + str(colorStatistics(backtestsignals['Sharpe'])) + "</font></b></li>"
          
    return ET.tostring(root, pretty_print =True)

def convertDateFormat(dateinput):
    date=time.strptime(str(dateinput), "%Y-%m-%d")
    day = date.tm_mday
    month = date.tm_mon
    year = date.tm_year
    date = str(month)+"/"+str(day)+"/"+str(year)
    return date

def colorStatistics(stats, roundFlag=True, forceRed = False):
    if stats < 0 or forceRed:
        color = dicoRGB['STATSRED']
    else:
        color = dicoRGB['STATSGREEN']
        
    if roundFlag==True:
        coloredStatistics = "<font color='"+color+"'>"+str(float(round((stats),2)))
    elif stats!='':
        coloredStatistics = "<font color='"+color+"'>"+str(insertComma(int(stats)))
    else:
        coloredStatistics = "<font color='"+color+"'>"
    return coloredStatistics

def insertComma(stats):
    if len(str(stats)) >= 4:
        commaIndex = len(str(stats))-4
        statsInsertComma = str(stats)[0:commaIndex+1]+","+str(stats)[commaIndex+1:]
    else:
        statsInsertComma = stats
    return statsInsertComma
    

def convertToSingleString(arrayOfStrings):
    endString = ''
    for item in arrayOfStrings:
        endString += item
    return endString


def timeSerieToXML(dates, values, title=None, legend=None, htitle=None, vltitle=None, vrtitle=None, axis=None, helpmsg=None, tooltip=None):
    def americanformat(datestring):
        y, m, d = datestring.split('-')
        return m+'/'+d+'/'+y

    xmlstring = '<chart'
    if title != None:
        xmlstring+= ' title="'+title+'"'
    if htitle != None:
        xmlstring+= ' horizontalAxisTitle="'+htitle+'"'
    if vltitle != None:
        xmlstring+= ' leftVerticalAxisTitle="'+vltitle+'"'
    if vrtitle != None:
        xmlstring+= ' rightVerticalAxisTitle="'+vrtitle+'"'
    xmlstring+= '>'

    if helpmsg != None:
        xmlstring+= '<HelpText><![CDATA['+helpmsg+']]></HelpText>'

    if tooltip != None:
        xmlstring+= '<ToolTip>'+tooltip[0]+'</ToolTip>'
        xmlstring+= '<DClickURL>'+tooltip[1]+'</DClickURL>'
    
    xmlstring+= '<series>'
    xmlstring+= '<dataseperator>;</dataseperator>'
    xmlstring+= '<renderer></renderer>'    
    if legend != None:
        xmlstring+= '<legend>'
        if type(legend) in [list, tuple]:
            xmlstring+= legend[0]
            for l in legend[1:]:
                xmlstring+= ', '+l
        else:
            xmlstring+= legend
        xmlstring+= '</legend>'
    xmlstring+= '<visible></visible>'

    if axis != None:
        if type(legend) in [list, tuple]:
            if len(legend) == len(axis):
                xmlstring+= '<axis>'+axis[0]
                for axe in axis[1:]:
                    xmlstring+= ','+axe
                xmlstring+='</axis>'
            else:
                raise ValueError('Legend and Axis should have the same number of elements')
        else:
            xmlstring+= '<axis>left</axis>'
    """else:
    if type(legend) in [list, tuple]:
        xmlstring+= '<axis>left,right</axis>'
    else:
        xmlstring+= '<axis>left</axis>'"""

    xmlstring+= '<datapoints>'
    for date, value in zip(dates, values):
        xmlstring+= americanformat(date)+', '
        if type(value) in [list, tuple]:
            xmlstring+= str(value[0])
            for v in value[1:]:
                xmlstring+= ', '+str(v)            
        else:
            xmlstring+= str(value)
        xmlstring+= ';\n'
    xmlstring+= '</datapoints>' 

    xmlstring+= '</series>'
    xmlstring+= '</chart>'

    return xmlstring

def updateDateMsg():
    now = datetime.datetime.now()
    today = now.date()
    return 'Last update on '+str(today.day)+'/'+str(today.month)+'/'+str(today.year)

def updateTimeMsg():
    now = datetime.datetime.now()
    today = now.date()
    strtime = str(now.hour)+'h'
    if now.minute<10:
        strtime+= '0'
    strtime+= str(now.minute)
    return 'Last update on '+str(today.day)+'/'+str(today.month)+'/'+str(today.year)+' at '+strtime

def main():

        print MakeXML()
        
        
if __name__=="__main__":
	main()
