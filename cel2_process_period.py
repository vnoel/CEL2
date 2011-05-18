#!/usr/bin/env python
# encoding: utf-8
"""
sel2_process_period.py
Created by Vincent Noel on 2010-11-26, LMD/CNRS.
"""

import sys
import os
import numpy as np
import sel2_orbit
import calipso

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
                    sel2_orbit.process_orbit(cal_file, with_cp=with_cp, replace=False)

def main():
    
    with_cp = True
    
    if len(sys.argv) < 2:
        print 'Usage : ./sel2_process_period.py PERIODE'
        print 'Example periods :'
        print ' 2009 - process all 2009'
        print ' 200903 - process all march 2009'
        print ' 20090310 - process all files from march 10 2009'
        print ' all - process 2006-2010'
        sys.exit(1)
    
    if len(sys.argv) > 2:
        if sys.argv[2] is 'DEBUG':
            sel2_orbit.debug = True
    
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

    print 'Running sel2_process_period.py'
    print '  - on years : ', years
    print '  - on months : ', months[0], '...', months[-1]
    print '  - on days : ', days[0], '...', days[-1]
    print '  - atb threshold : ', sel2_orbit.atb_min
    print '  - Working on temporary file copies : ', with_cp
    if sel2_orbit.debug:
        print '\n*** Debug run - only one orbit will be generated'
        print '*** diagnosis data will be saved in diagnose_orbit/ \n'
        if not os.path.isdir('diagnose_orbit/'):
            os.mkdir('diagnose_orbit')
            
    process_period(years, months, days, with_cp)

if __name__ == '__main__':
    main()
