#!/usr/bin/python2.5
import sys
import os
import datetime

def main(filenames, data_dir, dd, mm, yyyy):
    for filename in filenames:
        new_filename = filename + '.1'
        if os.path.exists(new_filename):
            os.remove(new_filename)
        if not os.path.exists(filename):
            print 'could not find %s.' %filename
            sys.exit(1)
        q_file = open(filename, 'r')
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
            if year == yyyy and date == day and month == mm:
                print 'ignoring %s in %s' %(q_tuple[:-1], filename)
            else:
                new_file = None
                if not os.path.exists(new_filename):
                    new_file = open(new_filename, 'w')
                else:
                    new_file = open(new_filename, 'a')
                new_file.write(q_tuple[:-1] + '\n')
                new_file.close()
        os.remove(filename)
        os.rename(new_filename, filename)
        print 'renamed %s to %s' %(new_filename, filename)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 4:
        print 'Usage: %s [data-dir] [dd] [mm] [yyyy]'
        sys.exit(1)
    data_dir = args[0]
    day = int(args[1])
    month = int(args[2])
    year = int(args[3])
    if not os.path.exists(data_dir):
        print '%s does not exist.' %data_dir
        sys.exit(1)
    all_pairs = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt') and f.startswith('$'):
            all_pairs.append(os.path.join(data_dir, f))
    main(all_pairs, data_dir, day, month, year)
    
