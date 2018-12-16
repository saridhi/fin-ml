
dicoRGB = {}
dicoRGB['ORANGE'] = "#F9B40D"
dicoRGB['RED'] = "#FF9999"
dicoRGB['DARKRED'] = "#FF6666"
dicoRGB['LIGHTBLUE'] = "#8CD9C9"
dicoRGB['BLUE'] = "#BBBBFF"
dicoRGB['DARKBLUE'] = "#6666FF"
dicoRGB['WHITE'] = "#FFFFFF"
dicoRGB['BLACK'] = "#000000"
dicoRGB['Bullish'] = dicoRGB['BLUE']
dicoRGB['Bearish'] = dicoRGB['ORANGE']
dicoRGB['Neutral'] = dicoRGB['WHITE']
dicoRGB['Trending'] = dicoRGB['BLUE']
dicoRGB['Ranging'] = dicoRGB['WHITE']
dicoRGB['Overbought'] = dicoRGB['ORANGE']
dicoRGB['Oversold'] = dicoRGB['BLUE']


#<HelpText><img src="totoro.jpg" height="100" width="80"></img>
#et reflagitate. moecha putida. <i>mutanda est ratio modusque uobis</i> siquid proficere 
from lxml import etree as ET
import datetime

def makeXML(title='This is a title', footer='This is a footer', keys=['First Col', 'Second Col'], table=[[0.,0.],[0.,0.]], bgcolors=None, helpmsg=None, helpimg=None, tooltip=None):
    import adaXMLParser as tXMLp
    portalXML = tXMLp.portalXMLParser()
    root = portalXML.createStructure()
    root.find(".//Title").text = title
    root.find(".//Footer").text = footer
    if tooltip != None:
        root.find(".//ToolTip").text = tooltip[0]
        root.find(".//DClickURL").text = tooltip[0]
    
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

def makeHeatMapXML(table=[[0.,0.],[0.,0.]], bgcolors=None):
    import adaXMLParser as tXMLp
    portalXML = tXMLp.portalXMLParser()
    root = portalXML.createStructureHM()
        
    if bgcolors == None:
        bgcolors = []
        for rowdata in table:
            rowcolor = ['WHITE' for data in rowdata]
            bgcolors.append(rowcolor)

    
        
    for rowdata, rowcolor in zip(table, bgcolors):
        row = ET.SubElement(root.find(".//Data"), "Row")
        for i in range(len(rowdata)):
            col = ET.SubElement(row, "Col"+str(i))
            if rowcolor[i][0] != '#':
                col.set("bgRGBColour", dicoRGB[rowcolor[i]])
            else:
                col.set("bgRGBColour", rowcolor[i])
            col.text = str(rowdata[i])
    

    return ET.tostring(root, pretty_print =True)


def makeSnapshotXML(indicatorresults=None,securities = ['EUR/GBP'],returnquickStats = [{}],returnflag = [{}]):
    import adaXMLParser as tXMLp
    portalXML = tXMLp.portalXMLParser()
    root = portalXML.createStructureSnapshot()
    
    columnsec = ET.SubElement(root.find(".//Columns"), "Column")
    columnsec.set("tagName",'Col0')
    displayname = ET.SubElement(columnsec, "DisplayName")
    displayname.text = 'Security'
    
    indicatorset = ['MA1','MA2','Cross','ADX','RSI','Stochastics','MACD','Doji','Engulfing','Hammer/SS','KRD']
    indicatorsetxml = ['MA1','MA2','MACrossover','ADX','RSI','Stochastics','MACD','Doji','Engulfing','Hammer/SS','KRD']

    for i in range(0,len(indicatorset)):
                column = ET.SubElement(root.find(".//Columns"), "Column")
                column.set("tagName", "Col"+str(i+1))
                displayname = ET.SubElement(column, "DisplayName")
                displayname.text = str(indicatorset[i])

    for sec in securities:
            row = ET.SubElement(root.find(".//Data"), "Row")
            colsec = ET.SubElement(row, "Col"+str(0))
            colsec.text = str(sec)

            
            
            indicatorresultssec = indicatorresults[securities.index(sec)]
            returnquickStatssec = returnquickStats[securities.index(sec)]
            flagsec = returnflag[securities.index(sec)]
            

            for indicators, indresults,quickstats,flag in zip(indicatorresultssec.keys(),indicatorresultssec.values(),returnquickStatssec.values(),flagsec.values()):

                    if (str(indresults) != ''):
                        col = ET.SubElement(row, "Col"+str(indicatorsetxml.index(indicators)+1))
                        col.text = str(indresults)
                    
                    
                        if str(flag) == 'Inactive':
                            col.set("bgRGBColour", dicoRGB['BLACK'])
                        else:
                            if indresults != 'Neutral':
                                col.set("bgRGBColour", dicoRGB[str(indresults)])

                    if (indicators == "Doji") or (indicators == "KRD") or (indicators == "Hammer/SS")or (indicators == "Engulfing"):
                        if quickstats != "":
                            col.set("Success",str(quickstats['Success']))
                            
                            col.set("Length",str(quickstats['Length'][0]))

                            col.set("Days",str(quickstats['Days'][0]))
                            
                            col.set("Returns",str(quickstats['Returns'][0]))

                            col.set("Reversal",str(quickstats['Reversal'][0]))

                            col.set("ExtremeVal",str(quickstats['ExtremeVal']))

                            col.set("CTMoves",str(quickstats['CTMoves']))
                    else:
                            
                        if quickstats != "":
                            
                            fulldates = quickstats['Dates']
                            lenfulldates = len(fulldates)
                            col.set("Dateofmostrecentsignal",str(quickstats['Dates'][lenfulldates-1]))
                        
                    

    return ET.tostring(root, pretty_print =True)

"""def makeSnapshotXML(indicatorresults=None,securities = ['EUR/GBP'],returnquickStats = [{}],returnflag = [{}]):
    import adaXMLParser as tXMLp
    portalXML = tXMLp.portalXMLParser()
    root = portalXML.createStructureSnapshot()
        
    for sec in securities:
            securityname = ET.SubElement(root.find(".//Data"), "Security")
            securityname.set("name",sec)
            indicatorresultssec = indicatorresults[securities.index(sec)]
            returnquickStatssec = returnquickStats[securities.index(sec)]
            flagsec = returnflag[securities.index(sec)]
            for indicators, indresults,quickstats,flag in zip(indicatorresultssec.keys(),indicatorresultssec.values(),returnquickStatssec.values(),flagsec.values()):

                    
                    if (indicators == 'MA1') or (indicators =='MA2') :
                        indicator = ET.SubElement(securityname, "IndicatorMA")
                    elif indicators == 'MACrossover':
                        indicator = ET.SubElement(securityname, "IndicatorCrossOver")
                    elif indicators == 'ADX':
                        indicator = ET.SubElement(securityname, "IndicatorADX")
                    elif indicators == 'RSI':
                        indicator = ET.SubElement(securityname, "IndicatorRSI")
                    elif indicators == 'Stochastics':
                        indicator = ET.SubElement(securityname, "IndicatorStochastics")
                    elif indicators == 'MACD':
                        indicator = ET.SubElement(securityname, "IndicatorMACD")
                    elif indicators == 'Doji':
                        indicator = ET.SubElement(securityname, "IndicatorDoji")
                    elif indicators == 'KRD':
                        indicator = ET.SubElement(securityname, "IndicatorKRD")
                    elif indicators == 'Hammer/SS':
                        indicator = ET.SubElement(securityname, "IndicatorHammer")
                    else:
                        indicator = ET.SubElement(securityname, "IndicatorEngulfing")
                    

                    indicator.set("value",str(indresults))
                    
                    color = ET.SubElement(indicator, "Color")
                    if (str(indresults) != ''):
                        if str(flag) == 'Inactive':
                            color.set("bgRGBColour", dicoRGB['BLACK'])
                        else:
                            color.set("bgRGBColour", dicoRGB[str(indresults)])
                    quickStats = ET.SubElement(indicator, "QuickStats")
                    
                    if (indicators == "Doji") or (indicators == "KRD") or (indicators == "Hammer/SS")or (indicators == "Engulfing"):
                        if quickstats != "":
                            success = ET.SubElement(quickStats, "Success")
                            success.set("value",str(quickstats['Success']))

                            length = ET.SubElement(quickStats, "Length")
                            length.set("value",str(quickstats['Length']))

                            days = ET.SubElement(quickStats, "Days")
                            days.set("value",str(quickstats['Days']))

                            returns = ET.SubElement(quickStats, "Returns")
                            returns.set("value",str(quickstats['Returns']))

                            reversal = ET.SubElement(quickStats, "Reversal")
                            reversal.set("value",str(quickstats['Reversal']))

                            extremeval = ET.SubElement(quickStats, "ExtremeVal")
                            extremeval.set("value",str(quickstats['ExtremeVal']))

                            ctmoves = ET.SubElement(quickStats, "CTMoves")
                            ctmoves.set("value",str(quickstats['CTMoves']))
                        else:
                            quickStats.set("value","")
                    else:
                            
                        if quickstats != "":
                            dates = ET.SubElement(quickStats, "Dateofmostrecentsignal")
                            fulldates = quickstats['Dates']
                            lenfulldates = len(fulldates)
                            dates.set("value",str(quickstats['Dates'][lenfulldates-1]))
                        else:
                            quickStats.set("value","")

    return ET.tostring(root, pretty_print =True)"""

def makeBacktestXML(returnbacktestStats = [{}]):
    import adaXMLParser as tXMLp
    portalXML = tXMLp.portalXMLParser()
    root = portalXML.createStructureBacktest()

    backtestavgdaily = returnbacktestStats['BacktestAvgDailyReturns']
    backtestsignals = returnbacktestStats['BacktestSignalsReturns']

    
    for element in root.iter("BacktestAvgDailyReturns"):
        element.find(".//Dates").text = str(backtestavgdaily['Dates'])
        element.find(".//Returns").text = str(backtestavgdaily['Returns'])
        element.find(".//Success").text = str(backtestavgdaily['Success'])
        element.find(".//Reversal").text = str(backtestavgdaily['Reversal'])
        element.find(".//MeanReturnoverDates").text = str(backtestavgdaily['MeanReturnoverDates'])
        element.find(".//MedianReturnoverDates").text = str(backtestavgdaily['MedianReturnoverDates'])
        element.find(".//MaxReturnoverDates").text = str(backtestavgdaily['MaxReturnoverDates'])
        element.find(".//MinReturnoverDates").text = str(backtestavgdaily['MinReturnoverDates'])
        element.find(".//SumReturnoverDates").text = str(backtestavgdaily['SumReturnoverDates'])
        element.find(".//StDevReturns").text = str(backtestavgdaily['StDevReturns'])
        element.find(".//ProbabilityPositiveReturn").text = str(backtestavgdaily['Probability Positive Return'])

    for element in root.iter("BacktestSignalsReturns"):
        element.find(".//Dates").text = str(backtestsignals['Dates'])
        element.find(".//Returns").text = str(backtestsignals['Returns'])
        element.find(".//Success").text = str(backtestsignals['Success'])
        element.find(".//Reversal").text = str(backtestsignals['Reversal'])
        element.find(".//MeanReturnoverDates").text = str(backtestsignals['MeanReturnoverDates'])
        element.find(".//MedianReturnoverDates").text = str(backtestsignals['MedianReturnoverDates'])
        element.find(".//MaxReturnoverDates").text = str(backtestsignals['MaxReturnoverDates'])
        element.find(".//MinReturnoverDates").text = str(backtestsignals['MinReturnoverDates'])
        element.find(".//SumReturnoverDates").text = str(backtestsignals['SumReturnoverDates'])
        element.find(".//StDevReturns").text = str(backtestsignals['StDevReturns'])
        element.find(".//ProbabilityPositiveReturn").text = str(backtestsignals['Probability Positive Return'])
        element.find(".//Investment").text = str(backtestsignals['Investment'])
        element.find(".//PL").text = str(backtestsignals['PL'])
        element.find(".//Sharpe").text = str(backtestsignals['Sharpe'])
        element.find(".//MaxDrawdown").text = str(backtestsignals['MaxDrawdown'])
        element.find(".//EndReturn").text = str(backtestsignals['EndReturn'])
    
    return ET.tostring(root, pretty_print =True)

    

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
