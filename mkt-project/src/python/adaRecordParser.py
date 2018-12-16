#!/usr/pin/python2.5
import os
import sys
import getopt
import numpy

from historicTrends import SuperModelRecord

def usage():
    print 'Usage: adaRecordParser.py --records=[records-csv] --output=[output-csv] [optional args]\n' + \
        '\trecords: a csv file containing all historical records.\n' + \
        '\toutput: a csv file containing the output records.\n' + \
        '\toptional args:\n' + \
        '\t\tm: pick only best and worst from each day\n' + \
        '\t\tw: output in weka-format\n' + \
        '\t\tp: append profit per trade as a percentage at the end of each row. do not use with -w'

def ReformatWeka(orig_str):
    all_parts = orig_str.split(',')
    todays_price = all_parts[2]
    next_price = all_parts[3]
    parts_one = all_parts[4:5]
    parts_two = all_parts[6:]
    verdict = ''
    if all_parts[5] == "Prediction_Today":
        verdict = "Prediction_Today"
    else:
        verdict = 'Fizz'
        if float(todays_price) > float(next_price):
            verdict = 'Spill'
    return all_parts[0] + ',' + ','.join(parts_one) + ',' + ','.join(parts_two) + ',' + verdict

def main():
    record_csv = None
    output_csv = None
    best_worst = False
    append_profit = False
    weka_format = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:o:mpw", 
                                   ["help", "records=", "output="])
    except getopt.GetoptError, err:
        print str(err) 
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
            output_csv = a
        elif o in ("-r", "--records"):
            record_csv = a
        elif o == "-m":
            best_worst = True
        elif o == "-p":
            append_profit = True
        elif o == "-w":
            weka_format = True
        else:
            assert False, "unhandled option, " + o

    if output_csv is None or record_csv is None:
        print 'Missing arguments. Try -h for help.'
        sys.exit(2)

    all_records = {}
    file = open(record_csv, 'r')
    first = True
    count = 0
    for row in file:
        if first:
            first = False
            continue
        count = count + 1
        row_stripped = row.rstrip('\n')
        parts = row_stripped.split(',')
        smr = SuperModelRecord(parts[0])
        smr.FromCSVString(row_stripped)
        all_records[parts[0] + "_" + parts[1]] = smr

    print 'there were %d records' %(count)
    print 'uniqued %d records' %(len(all_records))

    if best_worst:
        best_worst_records = {}
        for e in all_records:
            ep = e.split('_')
            fizz = True
            if all_records[e].score < 0.0:
                fizz = False
            hstr = ep[1] + '_' + str(fizz)
            if not hstr in best_worst_records:
                best_worst_records[hstr] = all_records[e]
            elif abs(all_records[e].score) > abs(best_worst_records[hstr].score):
                best_worst_records[hstr] = all_records[e]
        all_records = best_worst_records
        print 'best-worst filter: %d records remain' %(len(all_records))
    
    if os.path.exists(output_csv):
        print 'file: %s already exists. delete and re-run' %output_csv
        sys.exit(2)
    out_file = open(output_csv, 'w')

    if append_profit and weka_format:
        print 'Incompatible arguments. Weka format cannot have profit appended.'
        sys.exit(2)

    first = True
    mean_profit = 0.0
    var_profit = 0.0
    count = 0
    all_profits = []

    for e in all_records:            
        str_to_write = all_records[e].ToCSVString()
        count = count + 1
        if first:
            header_str = all_records[e].SMRHeader()
            if weka_format:
                header_str = ReformatWeka(header_str)
            out_file.write(header_str)
            if append_profit:
                out_file.write(',Profit')
            out_file.write('\n')
            first = False
        if weka_format:
            str_to_write = ReformatWeka(str_to_write)
        if append_profit:
            str_to_write += ',' + all_records[e].analString()
            all_profits.append(all_records[e].profit_perc)
            mean_profit += all_records[e].profit_perc
            var_profit += all_records[e].profit_perc * all_records[e].profit_perc
        out_file.write(str_to_write + '\n')
    out_file.close()
    print 'wrote', output_csv
    if append_profit:
        mean_profit /= float(count)
        var_profit /= float(count)
        var_profit -= mean_profit * mean_profit
        print '%.4f percent avg. trade profit. variance %.4f' %(mean_profit * 100.0, var_profit * 100.0)
        (val, edges) = numpy.histogram(all_profits, bins=40, range =(-0.08, 0.08))
        print 'Histogram of returns (profit percentage -> frequency)'
        for i in range(len(val)):
            print '[%.5f, %.5f]\t%d' %(edges[i] * 100, edges[i+1] * 100, val[i])

if __name__ == "__main__":
    main()
    
