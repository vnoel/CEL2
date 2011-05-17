#!/usr/bin/env python
# encoding: utf-8

"""

cel2_orbit.py

Created by Vincent Noel on 2011-05-17.

"""

import sys
import numpy as np
import calipso
import os
from scipy import ndimage

import cel2
import cel2_utils
import cel2_f

sys.setrecursionlimit(3000)

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
            
    sel2_data = sel2.sel2_data(cal.year, cal.month, cal.day, cal.orbit, cal_file)

    if not replace:
        # if the file already exists and replacing is not requested
        if os.path.isfile(sel2_data.filepath + '/' + sel2_data.filename):
            print 'File already exists : ', sel2_data.filepath+'/'+sel2_data.filename
            return

    lon, lat = cal.coords(navg=1)
    nprof = lon.size
    sel2_data.init_data(nprof)
    
    alt = calipso.lidar_alt
    
    time = cal.time(navg=1)
    atb = cal.atb(navg=1)
    atb1064 = cal.atb1064(navg=1)
    perp = cal.perp(navg=1)
    para = atb532 - perp
    temp = cal.temperature_on_lidar_alt(navg=1)
    mol = cal.mol_on_lidar_alt_calibrated(navg=1, zcal=(34,38))
    tropoz = cal.tropopause_height(navg=1)
    elev = cal.surface_elevation(navg=1)
    datatype = cal.orbit[-2:]
    cal.close()
    
    debug_print('Loaded %d profiles' % (atb.shape[0]))
    if debug:
        np.savez('debug_data/data_step1_avg.npz', lat=lat, lon=lon, alt=alt, atb=atb532, mol=mol, temp=temp)
        matlab.savemat('diagnose_orbit/data_step1_avg.mat', dict(lat=lat, lon=lon, alt=alt, atb=atb532, mol=mol, temp=temp))

    # check latitude continuity
    debug_print('Checking latitude continuity')
    dlat = np.abs(np.diff(lat))
    idx = (dlat > 0.1)
    atb532[idx,:] = -9999.
    debug_print('Found %d problems' % (idx.sum()))

    ground_return = compute_ground_return(atb532, alt, elev)

    # filters
    # TODO: need to find a more intelligent framework for this
    debug_print('Filtering data')

    # removes obviously unphysical values
    idx = (atb532 > 0.1) | (atb532 < -0.1)
    atb532[idx] = -9999.
    
    idx = (mol > 0.1) | (mol < -0.1)
    mol[idx] = -9999.
    
    idx = (perp < -0.1) | (perp > 0.1)
    perp[idx] = -9999.
    
    idx = (atb1064 > 0.1) | (atb1064 < -0.1)
    atb1064[idx] = -9999.
    
    idx = (temp < -120) | (temp > 80)
    temp[idx] = -9999.

    # removes below-ground signal
    atb532 = atb_remove_below_ground(atb532, alt, elev)

    # removes low snr
    snr = atb_compute_snr(atb532, alt)
    atb532 = data_remove_low_snr(atb532, alt, snr, datatype)
    temp = data_remove_low_snr(temp, alt, snr, datatype)
    
    if debug:
        np.savez('diagnose_orbit/data_step2_snr.npz', lon=lon, lat=lat, alt=alt, atb=atb532)
        matlab.savemat('diagnose_orbit/data_step2_snr.mat', dict(lon=lon, lat=lat, alt=alt, atb=atb532))
    
    # detect clouds from atb and molecular difference
    base, top = detect_clouds_from_atb(atb532, mol, alt, datatype)
    
    # merge close layers
    base, top = merge_close_layers(base, top)
    
    # removes ground return from layers 
    base, top = remove_layers_below_ground(base, top, elev)
        
    # check for opaque layers
    opacity = layers_opacity(base, top, ground_return, datatype)
    
    # layers cleaning
    base, top = clouds_remove_1km_above_tropopause(base, top, tropoz)
    base, top = clouds_remove_short_layers(base, top)
    
    # find individual clouds with horizontal extension,
    # attribute cloud ID to layers
    cloud_id = sel2_utils.create_cloud_id(base, top)
    hext = sel2_utils.find_cloud_horizontal_extension(cloud_id)

    # averages layer temperatures
    ltemp = compute_layer_temp(base, top, temp, alt)

    if debug:
        # create cloud mask
        cmask = np.zeros_like(atb532)
        for i in np.r_[0:base.shape[0]]:
            for j in np.r_[0:base.shape[1]]:
                if base[i,j,] > 0 and top[i,j] > 0:
                    idx = (alt >= base[i,j]) & (alt < top[i,j])
                    cmask[i,idx] = 1
        cloud_id, nclouds = ndimage.label(cmask > 0)
        np.savez('diagnose_orbit/data_step3_cmask.npz', lon=lon, lat=lat, alt=alt, cloudmask=cmask, cloud_id=cloud_id)
        matlab.savemat('diagnose_orbit/data_step3_cmask.mat', dict(lon=lon, lat=lat, alt=alt, cloudmask=cmask, cloud_id=cloud_id) )
                        
    sel2_data.set_time(time)
    sel2_data.set_temperature(ltemp)
    sel2_data.set_coords(lon, lat)
    sel2_data.set_layers(base, top)
    sel2_data.set_opacity_flag(opacity)
    sel2_data.set_cloud_id(cloud_id)
    sel2_data.set_horizontal_extension(hext)

    debug_print('Computing cloud layer properties')

    # Compute layer properties
    iatb = compute_layer_iatb(base, top, atb532, alt)
    od = compute_optical_depth(iatb)
    vdp = compute_layer_volume_depolarization(base, top, para, perp, alt)
    pdp, part_para, part_perp = compute_layer_particulate_depolarization(base, top, para, perp, alt, mol)
    vcr = compute_layer_volume_color_ratio(base, top, atb532, atb1064, alt)
    pcr = compute_particulate_color_ratio(base, top, atb532, atb1064, alt, mol)
    
    if debug:
        # find interesting values for each identified cloud
        # maximum optical depth and thickness for each cloud
        c_max_od = np.ones([nclouds+1]) * -9999.
        c_max_dz = np.ones([nclouds+1]) * -9999.
        for i in np.r_[0:base.shape[0]]:
            for j in np.r_[0:base.shape[1]]:
                if base[i,j] < 0 or top[i,j] < 0:
                    continue
                idx = (alt > base[i,j]) & (alt < top[i,j])
                cid = np.max(cloud_id[i,idx])
                if c_max_od[cid] < od[i,j]:
                    c_max_od[cid] = od[i,j]
                
                cdz = top[i,j] - base[i,j]
                if c_max_dz[cid] < cdz:
                    c_max_dz[cid] = cdz
                                        
        # horizontal size
        sls = ndimage.find_objects(cloud_id)
        c_hext = np.zeros([nclouds+1]) 
        for i, sl in enumerate(sls):
            hsl, vsl = sl
            c_hext[i] = (hsl.stop - hsl.start) * 5.

        # average temperature
        c_avg_t = np.zeros([nclouds+1])
        for i in np.r_[0:nclouds+1]:
            idx = (cloud_id == i) & (temp > -200)
            c_avg_t[i] = np.mean(temp[idx])

        np.savez('diagnose_orbit/data_step4_cprop.npz', c_max_od=c_max_od, c_max_dz=c_max_dz, c_hext=c_hext)
        matlab.savemat('diagnose_orbit/data_step4_cprop.mat', dict(c_max_od=c_max_od, c_max_dz=c_max_dz, c_hext=c_hext))
    
    sel2_data.set_iatb(iatb)
    sel2_data.set_od(od)
    sel2_data.set_volume_depolarization(vdp)
    sel2_data.set_particulate_depolarization(pdp)
    sel2_data.set_particulate_parallel_backscatter(part_para)
    sel2_data.set_particulate_perpendicular_backscatter(part_perp)
    sel2_data.set_volume_color_ratio(vcr)
    sel2_data.set_particulate_color_ratio(pcr)
    
    if debug:
        # would be nice to write those in a orbit-specific file
        idx = base > 0
        debug_print('Found %d layers' % (idx.sum()))
        debug_print(' = %f layer / profile' % (1. * idx.sum() / atb532.shape[0]))
        thickness = np.mean(top[idx] - base[idx])
        debug_print('Average thickness : %5.2f km' % (thickness))
        debug_print('Average base : %5.2f km' % (np.mean(base[idx])))
        debug_print('Average top : %5.2f km' % (np.mean(top[idx])))
        idx = od > 0
        debug_print('Average optical thickness : %5.2f' % (np.mean(od[idx])))
        idx = vdp > 0
        debug_print('Average volume depol : %5.2f' % (np.mean(vdp[idx])))
        idx = pdp > 0
        debug_print('Average particulate depol : %5.2f' % (np.mean(pdp[idx])))
    
    sel2_data.invalid_data_based_on(base)
    sel2_data.save()

    if with_cp:
        cmd = 'rm -f /scratch2/noel/' + filename
        os.system(cmd)


    
def main():

    if len(sys.argv) < 2:
        print 'Usage : ./sel2_orbit.py calipso_l1_file'
        print '(eventually followed by DEBUG)'
        sys.exit(1)
        
    cal_file = sys.argv[1]
    if len(sys.argv) > 2:
        sel2.debug = (sys.argv[2] == 'DEBUG')
        
    process_orbit(cal_file, with_cp=False, debug=sel2.debug)

if __name__ == '__main__':
    main()

