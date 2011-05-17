#!/usr/bin/env python
# encoding: utf-8
"""
test_read_speed.py
Created by Vincent Noel on 2010-11-26, LMD/CNRS.
"""

import cProfile
import pstats
import os

def main(cal_file, with_cp):

    from pyhdf.SD import SD

    if with_cp:
        cmd = 'cp %s /home/noel/scratch/' % (cal_file)
        print "running "+cmd
        os.system(cmd)
        filename = os.path.basename(cal_file)
        cal_file = '/home/noel/scratch/' + filename
                        
    print 'Reading ' + cal_file 
    
    vars = ['Latitude', 'Longitude', 
            'Total_Attenuated_Backscatter_532', 'Attenuated_Backscatter_1064', 'Perpendicular_Attenuated_Backscatter_532',
            'Pressure', 'Temperature', 'Molecular_Number_Density', 'Tropopause_Height', 'Surface_Elevation']
    
    hdf = SD(cal_file)
    for var in vars:
        print 'Reading ' + var
        hdf_var = hdf.select(var)
        data = hdf_var.get()
        hdf_var.endaccess()
    hdf.end()
    
    print 'ok.'
    if with_cp:
        print 'Removing '+filename
        cmd = 'rm -f /home/noel/scratch/' + filename
        os.system(cmd)
    

if __name__ == '__main__':
    file_warmup = '/DATA/LIENS/CALIOP/CAL_LID_L1.v3.01/2009/2009_01_01/CAL_LID_L1-ValStage1-V3-01.2009-01-01T21-33-20ZN.hdf'
    
    # compare reading times for the following identical files in different locations
    file1 = '/home/noel/scratch/CAL_LID_L1-ValStage1-V3-01.2009-01-01T00-07-47ZN.hdf'
    file2 = '/DATA/LIENS/CALIOP/CAL_LID_L1.v3.01/2009/2009_01_01/CAL_LID_L1-ValStage1-V3-01.2009-01-01T00-07-47ZN.hdf'
    file3 = '/DATA/FS105/CALIOP/CAL_LID_L1.v3.01/2009/2009_01_01/CAL_LID_L1-ValStage1-V3-01.2009-01-01T00-07-47ZN.hdf'
    
    # perform a dummy warm-up run
    print 'Performing warm-up run'
    cProfile.run('main("%s", False)' % (file_warmup) , 'test_read_speed')

    files = [file2, file3]
    with_cps = [False, True]
    for with_cp in with_cps:
        for file in files:
            cProfile.run('main("%s", %d)' % (file, with_cp) , 'test_read_speed')
            p = pstats.Stats('test_read_speed')
            p.sort_stats('time').print_stats(10)
