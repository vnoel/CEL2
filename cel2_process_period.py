#!/usr/bin/env python
# encoding: utf-8

"""
cel2_process_period.py
Created by Vincent Noel on 2011-05-18
"""

import sys
import os
import numpy as np
import cel2_orbit
import cel2
import calipso

cel2.outpath = '/homedata/noel/Projects/CEL2/'

def process_period(years, months, days, with_cp):

    for year in years:
        for month in months:
            for day in days:
                cal_list = calipso.l1_night_files(year, month, day)
                print '%04d-%02d-%02d - %d files' % (year, month, day, len(cal_list))
                if not cal_list:
                    continue
                    
                cal_list.sort()
                for cal_file in cal_list:
                    cel2_orbit.process_orbit_file(cal_file, with_cp=with_cp, replace=True)

def main():
    
    with_cp = False
    
    if len(sys.argv) < 2:
        print 'Usage : ./cel2_process_period.py PERIODE'
        print 'Example periods :'
        print ' 2009 - process all 2009'
        print ' 200903 - process all march 2009'
        print ' 20090310 - process all files from march 10 2009'
        print ' all - process 2006-2010'
        sys.exit(1)
    
    periode = sys.argv[1]

    years = np.r_[2006:2011]
    months = np.r_[1:13]
    days = np.r_[1:32]

    if len(periode) >= 4:
        years = [int(periode[0:4])]
    
    if len(periode) >= 6:
        months = [int(periode[4:6])]
    
    if len(periode) is 8:
        days = [int(periode[6:8])]

    print 'Running cel2_process_period.py'
    print '  - years : ', years
    print '  - months : ', months[0], '...', months[-1]
    print '  - days : ', days[0], '...', days[-1]
    print '  - Working on temporary file copies : ', with_cp
    print '  - output directory :', cel2.outpath
            
    process_period(years, months, days, with_cp)

if __name__ == '__main__':
    main()
