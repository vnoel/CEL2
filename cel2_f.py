
# fonctions utilisees par cel2_orbit.py

import numpy as np
from scipy.integrate import trapz
from scipy import ndimage

# the threshold values below have been carefully calibrated for
# nighttime data, not so much for daytime

# minimum snr for alt > 12 km
min_snr_high = {'ZN':1.5, 'ZD':1.4}

# minimum snr for 8.2 < alt < 12 km
min_snr_mid = {'ZN':4, 'ZD':9}

# minimum snr for alt < 8.2 km
min_snr_low = {'ZN':10, 'ZD':9}

# max number of layers
nl = 30

# particular atb threshold for cloud detection
atb_min = {'ZN':5e-4, 'ZD':1e-5}


def _integrate_signal(data, alt):
    '''
    Integrate signal as a function of altitude
    '''
    
    # need to do better than this !
    if data.size < 2:
        return data
    
    # we need to invert the order of items since altitude vector is top to bottom
    # (otherwise the integration is negative)
    integrate = trapz(data[::-1], x=alt[::-1])
    return integrate


def data_compute_snr(atb, alt):
    # used in cel2_orbit.py
    
    idx = (alt >= 28) & (alt <= 30)
    noise = np.std(atb[:,idx], axis=1)
    noise = np.tile(noise, [alt.size, 1]).T
    
    # noise is shape (nprof, nalt)
    snr = atb / noise
    
    return snr


def data_remove_below(atb, alt, zvector, invalid=-9999.):
    # used in cel2_orbit.py

    nprof = atb.shape[0]
    for iprof in np.r_[0:nprof]:
        idx = alt <= zvector[iprof]
        atb[iprof,idx] = invalid

    return atb

    
def data_remove_low_snr(data, alt, snr, datatype, invalid=-9999.):
    # used in cel2_orbit.py
    
    snr_thresh = np.ones(data.shape)
    idx = (alt < 8.185) 
    snr_thresh[:,idx] = min_snr_low[datatype]
    idx = (alt >= 8.185) & (alt < 12)
    snr_thresh[:,idx] = min_snr_mid[datatype]
    idx = (alt >= 12)
    snr_thresh[:,idx] = min_snr_high[datatype]
    
    data[snr < snr_thresh] = invalid
    
    return data


def _cloud_mask_remove_size(cloud_mask, alt, horizontal_resolution=0.333, hext_min=0.333, hext_max=1000000., vext_min=0.6, vext_max=100., clear_sky=0):
    
    labeled, nclouds = ndimage.label(cloud_mask)
    
    sls = ndimage.find_objects(labeled)
    for i, sl in enumerate(sls):
        hsl, vsl = sl
        hext = (hsl.stop - hsl.start) * horizontal_resolution
        vext = np.abs(alt[vsl.stop] - alt[vsl.start])
        if vext < vext_min or hext < hext_min or vext > vext_max or hext > hext_max:
            cloud_mask[sl] = clear_sky
    
    return cloud_mask


def _cloud_mask_find_layers(cloud_mask, alt, empty=-9999.):
    
    nprof, nalt = cloud_mask.shape
    base = np.ones([nprof, nl]) * empty
    top = np.ones([nprof, nl]) * empty
    cloud_mask_diff = np.diff(1.*cloud_mask, axis=1)
    for i in np.r_[0:nprof]:

        # cloud mask profile = 0 if clear-sky, 1 if cloud
        # diff(cloud mask) = 1 at cloud top, -1 at cloud base
        # altitude is top to bottom

        profdiff = cloud_mask_diff[i,:]
        idx_base = (profdiff < 0)
        idx_top = (profdiff > 0)
        n = np.min([idx_base.sum(), idx_top.sum(), nl])
        base[i,:n] = alt[idx_base][:n]
        top[i,:n] = alt[idx_top][:n]

    return base, top
        
            
def atb_find_layers(atb, mol, alt, datatype='ZN', hext_min=0.5, hext_max=1000000., vext_min=0.6, vext_max=100., invalid=-9999., clear_sky=-9998.):
    # used in cel2_orbit.py

    threshold = atb_min[datatype]

    # approximate particulate atb
    atb_part = atb - mol
    idx = (atb == invalid) | (mol == invalid)
    atb_part[idx] = invalid
    np.savez('debug_data/data_step2.4_atbpart.npz', atbpart=atb_part, mol=mol)

    atb_part[atb_part < threshold] = clear_sky
    cloud_mask = (atb_part > 0)
    np.savez('debug_data/data_step2.7_cmask.npz', cmask=cloud_mask)
    
    cloud_mask = _cloud_mask_remove_size(cloud_mask, alt, hext_min=hext_min, hext_max=hext_max, vext_min=vext_min, vext_max=vext_max, clear_sky=0)
    base, top = _cloud_mask_find_layers(cloud_mask, alt)

    return base, top


def layers_merge_close(base, top, closeness=0.12):

    nprof = base.shape[0]
    for i in np.arange(nprof):
        for j in np.arange(nl-1):
            if base[i,j] < -1000:
                continue

            if (base[i,j] - top[i,j+1]) <= closeness:
                base[i,j] = base[i,j+1]

                if j < (nl-1):
                    # offset the rest of the layers
                    base[i,j+1:nl-1] = base[i,j+2:nl]
                    top[i,j+1:nl-1] = top[i,j+2:nl]

                base[i,nl-1] = -9999.
                top[i,nl-1] = -9999.

    return base, top
    
    
def layers_remove_below(base, top, zvector, invalid=-9999.):
    
    nprof = base.shape[0]
    for i in np.arange(nprof):
        for j in np.arange(nl):
            if base[i,j] < -1000:
                continue
            if top[i,j] <= zvector[i]:
                base[i,j] = invalid
                top[i,j] = invalid
            if base[i,j] <= zvector[i]:
                base[i,j] = zvector[i]
            
    return base, top
    

def layers_remove_above(base, top, zvector, invalid=-9999.):
    nprof = base.shape[0]
    for i in np.r_[0:nprof]:
        for j in np.r_[0:nl]:
            if base[i,j] > zvector[i]:
                base[i,j] = invalid
                top[i,j] = invalid

    return base, top

    
def layers_opacity(base, top, ground_return, datatype):
    
    nprof = base.shape[0]
    opacity = np.zeros_like(base)

    for i in np.arange(nprof):

        # do we have ground return ? if yes, skip the profile
        if ground_return[i] > 0.05:
            continue

        # otherwise find the lowest layer
        bases = base[i,:]
        bases[bases < -1000] = +9999.        
        ilayer = np.argmin(bases)    
        if bases[ilayer] == 9999:
            continue
            
        # and flag it as opaque
        opacity[i,ilayer] = 1
        
    return opacity
                

def layers_temperature(base, top, temp, alt):

    nprof = temp.shape[0]
    ltemp = np.ones_like(top) * -9999.

    for iprof in np.r_[0:nprof]:
        tprof = temp[iprof,:]
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue

            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (tprof > -273.)
            if idx.sum() > 0:
                ltemp[iprof,ilayer] = np.mean(tprof[idx])

    return ltemp
                
           
def layers_cloud_id(base, top, alt):
    
    nprof = base.shape[0]
    nalt = alt.size
    
    cloud_mask = np.zeros((nprof, nalt))
    for i in np.r_[0:nprof]:
        for j in np.r_[0:nl]:
            if base[i,j] < 0 or top[i,j] < 0:
                continue
            idx = (alt >= base[i,j]) & (alt < top[i,j])
            cloud_mask[i,idx] = 1

    labeled, nclouds = ndimage.label(cloud_mask > 0)
    cloud_id = np.ones((nprof, nl)) * -9999.

    for i in np.r_[0:nprof]:
        for j in np.r_[0:nl]:
            if base[i,j] < 0 or top[i,j] < 0:
                continue
            idx = (alt >= base[i,j]) & (alt < top[i,j])
            if idx.sum() == 0:
                continue
            cloud_id[i,j] = np.max(labeled[i,idx])
                
    return cloud_id, labeled
    
def cloud_horizontal_extension(cloud_labeled_mask, horizontal_resolution=0.333):
    
    nclouds = np.max(cloud_labeled_mask)
    sls = ndimage.find_objects(cloud_labeled_mask)
    hext = np.zeros([nclouds+1])
    for i, sl in enumerate(sls):
        hsl, vsl = sl
        hext[i] = (hsl.stop - hsl.start) * horizontal_resolution
    return hext
    
                
def compute_ground_return(atb, alt, elev):
    
    nprof = atb.shape[0]
    ground_return = np.zeros(nprof)
    
    for i in np.arange(nprof):
        idx = (alt > (elev[i] - 0.3)) & (alt < (elev[i] + 0.1)) & (atb[i,:] > -100.)
        ground_return[i] = _integrate_signal(atb[i,idx], alt[idx])
        
    return ground_return
    
            
def layers_iatb(base, top, atb, alt):
    
    nprof = atb.shape[0]
    iatb = np.ones_like(top) * -9999.
    
    for iprof in np.r_[0:nprof]:
        prof = atb[iprof,:]
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
            
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (prof > 0)
            if idx.sum() > 0:
                iatb[iprof, ilayer] = _integrate_signal(prof[idx], alt[idx])
            
    return iatb

        
def layers_volume_depolarization(base, top, para, perp, alt, invalid=-9999.):
    
    nprof = para.shape[0]
    depol = np.ones_like(base) * invalid
    
    for iprof in np.r_[0:nprof]:
        
        if not np.any(top[iprof,:] > 0):
            continue
        
        paraprof = para[iprof,:]
        perpprof = perp[iprof,:]
        
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (paraprof > 0) & (perpprof > 0)
            if idx.sum() > 0:
                intperp = _integrate_signal(perpprof[idx], alt[idx])
                intpara = _integrate_signal(paraprof[idx], alt[idx])
                if np.isfinite(intperp/intpara):
                    depol[iprof,ilayer] = intperp / intpara
                if not np.isfinite(intperp/intpara):
                    print intperp, intpara, intperp/intpara, perpprof[idx], paraprof[idx], idx
                depol[iprof,ilayer] = intperp / intpara
                
    return depol
                
def layers_particulate_depolarization(base, top, para, perp, alt, mol):
    
    molperp = 0.02 * mol / 1.02
    molpara = mol - molperp
    
    depol = np.ones_like(base) * -9999.
    part_para = np.ones_like(base) * -9999.
    part_perp = np.ones_like(base) * -9999.
    
    nprof = para.shape[0]
    for iprof in np.r_[0:nprof]:
        if not np.any(top[iprof,:] > 0):
            continue
            
        paraprof = para[iprof,:]
        perpprof = perp[iprof,:]
        molparaprof = molpara[iprof,:]
        molperpprof = molperp[iprof,:]
        
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt > base[iprof, ilayer]) & (alt <= top[iprof, ilayer]) & (paraprof > 0) & (perpprof > 0) & (molparaprof > 0) & (molperpprof > 0)
            if idx.sum() > 0:
                # Approximation for particulate depolarization ratio
                # See Martins et al. 2010 for details
                intperp = _integrate_signal(perpprof[idx], alt[idx])
                intpara = _integrate_signal(paraprof[idx], alt[idx])
                intmolperp = _integrate_signal(molperpprof[idx], alt[idx])
                intmolpara = _integrate_signal(molparaprof[idx], alt[idx])
                
                part_perp[iprof,ilayer] = (intperp - intmolperp)
                part_para[iprof,ilayer] = (intpara - intmolpara)
                depol[iprof,ilayer] = (intperp - intmolperp) / (intpara - intmolpara)
        
    return depol, part_para, part_perp
                
def layers_volume_color_ratio(base, top, atb532, atb1064, alt):
    
    nprof = atb532.shape[0]
    vcr = np.ones_like(base) * -9999.
    
    for iprof in np.r_[0:nprof]:
        if not np.any(top[iprof,:] > 0):
            continue
        
        prof532 = atb532[iprof,:]
        prof1064 = atb1064[iprof,:]
        
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof, ilayer]) & (alt <= top[iprof,ilayer]) & (prof532 > 0) & (prof1064 > 0)
            if idx.sum() > 0:
                int1064 = _integrate_signal(prof1064[idx], alt[idx])
                int532 = _integrate_signal(prof532[idx], alt[idx])
                vcr[iprof,ilayer] = int1064 / int532
    
    return vcr
    
def layers_particulate_color_ratio(base, top, atb532, atb1064, alt, mol):
    
    Smol = 8 * np.pi / 3.
    
    nprof = atb532.shape[0]
    pcr = np.ones_like(base) * -9999.
    for iprof in np.r_[0:nprof]:
        if not np.any(top[iprof,:] > 0):
            continue

        prof532 = atb532[iprof,:]
        prof1064 = atb1064[iprof,:]
        profmol = mol[iprof,:]

        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (prof532 > 0) & (prof1064 > 0) & (profmol > 0)
            if idx.sum() > 0:
                # layer molecular extinction
                alphamol = Smol * _integrate_signal(profmol[idx], alt[idx])
                # layer molecular transmission
                tmol = np.exp(-2. * alphamol)
                # formula for approximation of color ratio from Tao et al. 2008
                # See Martins et al. 2010 for details
                int1064 = _integrate_signal(prof1064[idx], alt[idx])
                int532 = _integrate_signal(prof532[idx], alt[idx])
                intmol = _integrate_signal(profmol[idx], alt[idx])
                pcr[iprof,ilayer] = int1064 / (int532/tmol - intmol)
            
    return pcr
            
    
def layers_optical_depth(iatb532):
    
    eta = 0.7
    S = 25
    
    od = np.ones_like(iatb532) * -9999.
    idx = iatb532 > 0.
    od[idx] = -0.5 / eta * np.log(1. - 2. * eta * S * iatb532[idx])
    idx = np.isnan(od)
    
    # when iatb532 > 1/(2*eta*S), the log gives nan
    # this happens often for totally attenuating layers
    od[np.isnan(od)] = -9999.
    
    return od
        
        
def recouvrement(base, top, bases, tops):
    '''
    Finds out for a layer characterized by base and top if there are overlapping layers
    in the lists bases tops
    The overlap must be bigger than min_overlap 
    '''
    
    # minimum overlap = 60 meters, CALIOP resolution in upper troposphere (> 8.2. km)
    # (ie one data point is not enough for overlap)
    
    # here the minimum valid overlap corresponds to the layer geometrical thickness
    # divided by 4 (i.e., 4 points minimum given that a layer can only be as thin as 4 points on the vertical)
        
    # bases and tops and vectors for a given profile
    # base and top are actual altitudes for a given layer of a given profile
    
    nlay = bases.shape[0]
    found = np.ones(nlay)
    
    for i in np.r_[0:nlay]:
    
        if tops[i] < 0:
            found[i] = 0
            continue

        min_thickness = np.min([top-base, tops[i]-bases[i]])
        min_overlap = (min_thickness) / 3.        
            
        if (bases[i] > (top - min_overlap)) or (tops[i] < (base + min_overlap)):
            found[i] = 0
            
    return found
    


def data_clouds(data, base, top, alt):
    nprof = data.shape[0]
    data2 = np.ones_like(data) * -9999.
    for iprof in np.r_[0:nprof]:
        for ilayer in np.r_[0:nl]:
            if top[iprof,ilayer] < 0:
                continue
            
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer])
            data2[iprof,idx] = data[iprof,idx]
            
    return data2
