#!/usr/bin/python2.5
import sys
import os
import datetime

def main(filename, data_dir):
    all_pairs = open(filename, 'r')
    for pair in all_pairs:
        filename = data_dir + '$' + pair[:-1] + '.txt'
        new_filename = filename + '.1'
        if os.path.exists(new_filename):
            os.remove(new_filename)
        if not os.path.exists(filename):
            print 'could not find %s.' %filename
            sys.exit(1)
        q_file = open(filename, 'r')
        print 'ignoring ',
        for q_tuple in q_file:
            if not q_tuple:
                continue
            s_date = q_tuple.split(',')[0]
            if not s_date:
                continue
            parts = s_date.split('/')
            year = int(parts[2])
            month = int(parts[1])
            date = int(parts[0])
            q_date = datetime.date(year, month, date)
            if q_date.weekday() == 5 or q_date.weekday() == 6:
                print 'weekend  %s' %s_date,
            else:
                new_file = None
                if not os.path.exists(new_filename):
                    new_file = open(new_filename, 'w')
                else:
                    new_file = open(new_filename, 'a')
                new_file.write(q_tuple[:-1] + '\n')
                new_file.close()
        print
        os.remove(filename)
        os.rename(new_filename, filename)
        print 'renamed %s to %s' %(new_filename, filename)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print 'Usage: %s [file containing all pairs] [data-dir]'
        sys.exit(1)
    all_pairs_file = args[0]
    data_dir = args[1]
    main(all_pairs_file, data_dir)
    
