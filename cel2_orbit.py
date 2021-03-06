#!/usr/bin/env python
# encoding: utf-8

"""

cel2_orbit.py

Created by Vincent Noel on 2011-05-17.

If this file is called directly on a calipso level1 file (instead of being imported),
it sets debug mode and prints out lots of stuff,
it disables the copy mode.

"""

import sys
import numpy as np
import calipso
import os

import cel2
import cel2_f

navg = 1
debug_path = '/homedata/noel/Projects/CEL2/debug_data/'

def process_orbit_file(cal_file, with_cp=False, replace=True, debug=False):
    '''
    Process the half-orbit Level 1 file cal_file and create the associated level 2 file.
    '''
    
    if with_cp:
        # this speeds things up a lot on icare.
        cmd = 'cp %s /scratch2/noel/' % (cal_file)
        os.system(cmd)
        filename = os.path.basename(cal_file)
        cal_file = '/scratch2/noel/' + filename
        
    print 'Processing file : %s' % cal_file
    cal = calipso.Cal1(cal_file)
            
    cel2_data = cel2.cel2_data(cal.year, cal.month, cal.day, cal.orbit, cal_file)

    if not replace:
        # if the file already exists and replacing is not requested
        if os.path.isfile(cel2_data.filepath + '/' + cel2_data.filename):
            print 'File already exists : ', cel2_data.filepath+'/'+cel2_data.filename
            return

    lon, lat = cal.coords(navg=navg)
    nprof = lon.size
    cel2_data.init_data(nprof)
    
    alt = calipso.lidar_alt
    datatype = cal.orbit[-2:]

    time = cal.time(navg=navg)
    atb = cal.atb(navg=navg)
    atb1064 = cal.atb1064(navg=navg)
    perp = cal.perp(navg=navg)
    para = atb - perp
    temp = cal.temperature_on_lidar_alt(navg=navg)
    mol = cal.mol_on_lidar_alt_calibrated(navg=navg, navgh=1000, zcal=(32,36))
    tropoz = cal.tropopause_height(navg=navg)
    elev = cal.surface_elevation(navg=navg)
    cal.close()
    
    if debug:
        print 'Loaded %d profiles' % (atb.shape[0])
        np.savez(debug_path+'data_step1_avg.npz', lat=lat, lon=lon, alt=alt, atb=atb, mol=mol, temp=temp)
        print 'Data cleanup'

    # check latitude continuity
    
    dlat = np.abs(np.diff(lat))
    idx = (dlat > 0.1)
    if idx.sum() > 0:
        atb[idx,:] = -9999.
        print 'Found %d latitude problems' % (idx.sum())

    ground_return = cel2_f.compute_ground_return(atb, alt, elev)

    # remove obviously unphysical values    
    idx = (atb > 0.1) | (atb < -0.1) | (mol > 0.1) | (mol < -0.1) | (perp < -0.1) | (perp > 0.1) | (atb1064 > 0.1) | (atb1064 < -0.1) | (temp < -120) | (temp > 80)
    atb[idx] = -9999.

    # remove below-ground signal
    atb = cel2_f.data_remove_below(atb, alt, elev, invalid=-9999.)

    # remove areas with low SNR in ATB
    snr = cel2_f.data_compute_snr(atb, alt)
    atb = cel2_f.data_remove_low_snr(atb, alt, snr, datatype, invalid=-9999.)

    # propagate data validity
    idx = (atb==-9999.)
    mol[idx] = -9999.
    perp[idx] = -9999.
    atb1064[idx] = -9999.
    temp[idx] = -9999.

    if debug:
        np.savez(debug_path+'data_step2_snr.npz', lon=lon, lat=lat, alt=alt, atb=atb)
        print 'Detecting layers'

    # layer detection and cleanup
    base, top = cel2_f.atb_find_layers(atb, mol, alt, datatype, hext_min=0.3, hext_max=10000000., vext_min=0.6, vext_max=100.)
    base, top = cel2_f.layers_merge_close(base, top)
    base, top = cel2_f.layers_remove_below(base, top, elev)
    base, top = cel2_f.layers_remove_above(base, top, tropoz + 3.)

    # layer identification
    cloud_id, cloud_labeled_mask = cel2_f.layers_cloud_id(base, top, alt)
    hext = cel2_f.cloud_horizontal_extension(cloud_labeled_mask)
    
    # now we don't do anymore filtering during the dataset production phase.
    # any additional filtering will need to be done during dataset analysis,
    # using the properties computed afterwards.

    if debug:
        np.savez(debug_path+'data_step3_cmask.npz', lon=lon, lat=lat, alt=alt, base=base, hext=hext, top=top, cloud_id=cloud_id, cloud_labeled_mask=cloud_labeled_mask)
        print 'Calculating layer properties'

    # layers properties

    opacity = cel2_f.layers_opacity(base, top, ground_return, datatype)
    ltemp = cel2_f.layers_temperature(base, top, temp, alt)
    iatb = cel2_f.layers_iatb(base, top, atb, alt)
    od = cel2_f.layers_optical_depth(iatb)
    vdp = cel2_f.layers_volume_depolarization(base, top, para, perp, alt)
    pdp, part_para, part_perp = cel2_f.layers_particulate_depolarization(base, top, para, perp, alt, mol)
    vcr = cel2_f.layers_volume_color_ratio(base, top, atb, atb1064, alt)
    pcr = cel2_f.layers_particulate_color_ratio(base, top, atb, atb1064, alt, mol)
                        
    cel2_data.set_time(time)
    cel2_data.set_temperature(ltemp)
    cel2_data.set_coords(lon, lat)
    cel2_data.set_layers(base, top)
    cel2_data.set_opacity_flag(opacity)
    cel2_data.set_cloud_id(cloud_id)
    cel2_data.set_horizontal_extension(hext)
    
    cel2_data.set_iatb(iatb)
    cel2_data.set_od(od)
    cel2_data.set_volume_depolarization(vdp)
    cel2_data.set_particulate_depolarization(pdp)
    cel2_data.set_particulate_parallel_backscatter(part_para)
    cel2_data.set_particulate_perpendicular_backscatter(part_perp)
    cel2_data.set_volume_color_ratio(vcr)
    cel2_data.set_particulate_color_ratio(pcr)
    
    if debug:
        idx = base > 0
        print 'Found %d layers' % idx.sum()
        print ' = %f layer / profile' % (1. * idx.sum() / atb.shape[0])
    
    cel2_data.invalid_data_based_on(base)
    cel2_data.save()

    if with_cp:
        cmd = 'rm -f /scratch2/noel/' + filename
        os.system(cmd)

    
def main():

    if len(sys.argv) < 2:
        print 'Usage : ./cel2_orbit.py calipso_l1_file'
        sys.exit(1)
        
    if not os.path.isdir(debug_path):
        os.mkdir(debug_path)
        
    cal_file = sys.argv[1]        
    process_orbit_file(cal_file, with_cp=False, debug=True)

if __name__ == '__main__':
    main()

