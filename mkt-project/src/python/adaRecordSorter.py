#!/usr/pin/python2.5
import os
import sys
import datetime

def main():
    args = sys.argv[1:]
    if not args or len(args) < 2:
        print 'Usage: %s [input csv-filename] [output csv-filename] [date-field-no (def 2)]'
        sys.exit(1)
    inp_name = args[0]
    out_name = args[1]
    date_field = 2
    if len(args) > 2:
        date_field = int(args[2])
    date_field = date_field - 1

    def compare_records(x, y):
        date_x = x.split(',')[date_field].split('/')
        date_y = y.split(',')[date_field].split('/')
        ddate_x = datetime.date(int(date_x[2]), int(date_x[1]), int(date_x[0]))
        ddate_y = datetime.date(int(date_y[2]), int(date_y[1]), int(date_y[0]))
        if ddate_x < ddate_y:
            return -1
        return 1

    if os.path.exists(out_name):
        print 'file %s exists. delete and re-run.' %out_name
        sys.exit(1)
    if not os.path.exists(inp_name):
        print 'file %s does not exist.' %inp_name
        sys.exit(1)
    inp_file = open(inp_name, 'r')
    first = True
    header = ''
    all_rows = []
    for row in inp_file:
        if first:
            header = row.rstrip('\n')
            first = False
            continue
        all_rows.append(row.rstrip('\n'))
    all_rows.sort(cmp = compare_records)
    out_file = open(out_name, 'w')
    out_file.write(header + '\n')
    for e in all_rows:
        out_file.write(e + '\n')
    print 'wrote file %s' %out_name
    out_file.close()

if __name__ == "__main__":
    main()
        
        
