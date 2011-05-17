
# fonctions utilisees par cel2_orbit.py

import numpy as np
from scipy.integrate import trapz
import sel2

# the threshold values below have been carefully calibrated for
# nighttime data, not so much for daytime

# minimum snr for alt > 12 km
min_snr_high = {'ZN':4.5, 'ZD':1.4}

# minimum snr for 8.2 < alt < 12 km
min_snr_mid = {'ZN':20, 'ZD':9}

# minimum snr for alt < 8.2 km
min_snr_low = {'ZN':35, 'ZD':9}


# max number of layers
nl = 20

# particular atb threshold for cloud detection
atb_min = {'ZN':5.2e-5, 'ZD':7e-5}

debug = sel2.debug
debug_file = sel2.debug_file

def debug_print(string):
    if debug:
        print(string)    

def integrate_signal(data, alt):
    '''
    Integrate signal as a function of altitude
    '''
    
    # we need to invert the order of items since altitude vector is top to bottom
    # (otherwise the integration is negative)
    integrate = trapz(data[::-1], x=alt[::-1])
    return integrate

def atb_compute_snr(atb, alt):
    
    idx = (alt >= 28) & (alt <= 30)
    noise = np.std(atb[:,idx], axis=1)
    noise = np.tile(noise, [alt.size, 1]).T
    
    # noise is shape (nprof, nalt)
    snr = atb / noise
    
    return snr
    
def data_remove_low_snr(data, alt, snr, datatype):
    
    # this is equivalent to the matlab function keepgoodsky.m
    
    snr_thresh = np.ones(data.shape)
    idx = (alt < 8.185) 
    snr_thresh[:,idx] = min_snr_low[datatype]
    idx = (alt >= 8.185) & (alt < 12)
    snr_thresh[:,idx] = min_snr_mid[datatype]
    idx = (alt >= 12)
    snr_thresh[:,idx] = min_snr_high[datatype]
    
    data[snr < snr_thresh] = -9999.
    
    return data

def data_remove_small_vertical_features(data, small=4):
    
    # this is equivalent to the matlab function nonoise.m
    
    nprof, nalt = data.shape
    
    for i in np.r_[0:nprof]:
        ialt = 1
        while ialt < nalt:
            idx = data[i,ialt:] > 0
            if np.sum(idx) == 0:
                break
            
            while data[i,ialt] <= 0 and ialt < nalt:
                ialt += 1
                
            # first valid data : ialt
            # look for the end of the valid data succession
            
            ialt2 = ialt
            while (data[i,ialt2] > 0) and (ialt2 < nalt):
                ialt2 += 1
            
            if (ialt2-ialt) < small:
                data[i,ialt:ialt2] = -9999.
            
            ialt = ialt2
            
    return data
    
def detect_clouds_from_part_atb(atb, alt):
    '''
    Detect cloud in cleaned-up particulate backscatter
    (i.e. atb < threshold and small features were removed previously)
    '''
    
    # this is equivalent to the matlab function DetectBottomTopCloud.m

    nprof, nalt = atb.shape
    base = np.ones([nprof, nl]) * -9999.
    top = np.ones_like(base) * -9999.
    
    for iprof in np.r_[0:nprof]:

        prof = atb[iprof,:]
        ialt = 0
        ilayer = 0
                
        while ialt < nalt and ilayer < nl:
            
            if np.all(np.isnan(prof[ialt:])):
                break
        
            # look for the first valid value
            while ialt < nalt:
                if prof[ialt] > 0:
                    break
                ialt += 1
                
            if ialt==nalt:
                break
                
            # first valid value = cloudtop
            ialttop = ialt
        
            # now look for the first invalid value
            while ialt < nalt:
                if (prof[ialt] < 0) and ((ialt-ialttop) > 3):
                    break
                ialt += 1
            ialtbase = ialt

            top[iprof,ilayer] = alt[ialttop]
            base[iprof,ilayer] = alt[ialtbase]
            
            ilayer += 1
            
    return base, top
    

def detect_clouds_from_atb(atb, mol, alt, datatype):
    '''
    Detect cloud using attenuated total backscatter and molecular backscatter
    '''

    # datatype = ZN or ZD
    threshold = atb_min[datatype]

    # approximate particulate atb
    atb_part = atb - mol
    idx = (atb < -10) | (mol < -10)
    atb_part[idx] = -9999.

    # clear-sky removal
    atb_part[atb_part < threshold] = -9999.

    debug_print('Removing small features')
    atb_part = data_remove_small_vertical_features(atb_part)

    debug_print('Detecting cloud layers')
    base, top = detect_clouds_from_part_atb(atb_part, alt)

    return base, top

def remove_layers_below_ground(base, top, elev):
    
    nprof = base.shape[0]
    for i in np.arange(nprof):
        for j in np.arange(nl):
            if base[i,j] < -1000:
                continue
                
            if top[i,j] <= elev[i]:
                base[i,j] = -9999.
                top[i,j] = -9999.

            if base[i,j] <= elev[i]:
                base[i,j] = elev[i]
                
            
    return base, top
    
def merge_close_layers(base, top, closeness=0.12):
    
    # the first detected cloud layer is the highest
    
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
                
def compute_ground_return(atb, alt, elev):
    
    nprof = atb.shape[0]
    ground_return = np.zeros(nprof)
    
    for i in np.arange(nprof):
        idx = (alt > (elev[i] - 0.3)) & (alt < (elev[i] + 0.1)) & (atb[i,:] > -100.)
        ground_return[i] = integrate_signal(atb[i,idx], alt[idx])
        
    return ground_return
    
            
def compute_layer_iatb(base, top, atb, alt):
        
    # this was before part of DetectBottomTopCloud.m
    
    nprof = atb.shape[0]
    iatb = np.ones_like(top) * -9999.
    
    for iprof in np.r_[0:nprof]:
        prof = atb[iprof,:]
        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
            
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (prof > 0)
            if idx.sum() > 0:
                iatb[iprof, ilayer] = integrate_signal(prof[idx], alt[idx])
            
    return iatb
    
def compute_layer_temp(base, top, temp, alt):

    # this was before part of DetectBottomTopCloud.m
    
    nprof = temp.shape[0]
    ltemp = np.ones_like(top) * -9999.
    
    for iprof in np.r_[0:nprof]:
        tprof = temp[iprof,:]
        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
            
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (tprof > -273.)
            if idx.sum() > 0:
                ltemp[iprof,ilayer] = np.mean(tprof[idx])

    return ltemp
        
def compute_layer_volume_depolarization(base, top, para, perp, alt):
    
    nprof = para.shape[0]
    depol = np.ones_like(base) * -9999.
    
    for iprof in np.r_[0:nprof]:
        
        if not np.any(top[iprof,:] > 0):
            continue
        
        paraprof = para[iprof,:]
        perpprof = perp[iprof,:]
        
        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (paraprof > 0) & (perpprof > 0)
            if idx.sum() > 0:
                intperp = integrate_signal(perpprof[idx], alt[idx])
                intpara = integrate_signal(paraprof[idx], alt[idx])
                depol[iprof,ilayer] = intperp / intpara
                
    return depol
                
def compute_layer_particulate_depolarization(base, top, para, perp, alt, mol):
    
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
        
        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt > base[iprof, ilayer]) & (alt <= top[iprof, ilayer]) & (paraprof > 0) & (perpprof > 0) & (molparaprof > 0) & (molperpprof > 0)
            if idx.sum() > 0:
                # Approximation for particulate depolarization ratio
                # See Martins et al. 2010 for details
                intperp = integrate_signal(perpprof[idx], alt[idx])
                intpara = integrate_signal(paraprof[idx], alt[idx])
                intmolperp = integrate_signal(molperpprof[idx], alt[idx])
                intmolpara = integrate_signal(molparaprof[idx], alt[idx])
                
                part_perp[iprof,ilayer] = (intperp - intmolperp)
                part_para[iprof,ilayer] = (intpara - intmolpara)
                depol[iprof,ilayer] = (intperp - intmolperp) / (intpara - intmolpara)
        
    return depol, part_para, part_perp
                
def compute_layer_volume_color_ratio(base, top, atb532, atb1064, alt):
    
    nprof = atb532.shape[0]
    vcr = np.ones_like(base) * -9999.
    
    for iprof in np.r_[0:nprof]:
        if not np.any(top[iprof,:] > 0):
            continue
        
        prof532 = atb532[iprof,:]
        prof1064 = atb1064[iprof,:]
        
        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof, ilayer]) & (alt <= top[iprof,ilayer]) & (prof532 > 0) & (prof1064 > 0)
            if idx.sum() > 0:
                int1064 = integrate_signal(prof1064[idx], alt[idx])
                int532 = integrate_signal(prof532[idx], alt[idx])
                vcr[iprof,ilayer] = int1064 / int532
    
    return vcr
    
def compute_particulate_color_ratio(base, top, atb532, atb1064, alt, mol):
    
    Smol = 8 * np.pi / 3.
    
    nprof = atb532.shape[0]
    pcr = np.ones_like(base) * -9999.
    for iprof in np.r_[0:nprof]:
        if not np.any(top[iprof,:] > 0):
            continue

        prof532 = atb532[iprof,:]
        prof1064 = atb1064[iprof,:]
        profmol = mol[iprof,:]

        for ilayer in np.r_[0:20]:
            if top[iprof,ilayer] < 0:
                continue
                
            idx = (alt >= base[iprof,ilayer]) & (alt <= top[iprof,ilayer]) & (prof532 > 0) & (prof1064 > 0) & (profmol > 0)
            if idx.sum() > 0:
                # layer molecular extinction
                alphamol = Smol * integrate_signal(profmol[idx], alt[idx])
                # layer molecular transmission
                tmol = np.exp(-2. * alphamol)
                # formula for approximation of color ratio from Tao et al. 2008
                # See Martins et al. 2010 for details
                int1064 = integrate_signal(prof1064[idx], alt[idx])
                int532 = integrate_signal(prof532[idx], alt[idx])
                intmol = integrate_signal(profmol[idx], alt[idx])
                pcr[iprof,ilayer] = int1064 / (int532/tmol - intmol)
            
    return pcr
            
    
def compute_optical_depth(iatb532):
    
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
    
def clouds_remove_short_layers(base, top):
    
    # this was the matlab function ClearSky4.m
    
    nprof, nlay = base.shape
    
    base2 = np.ones_like(base) * -9999.
    top2 = np.ones_like(top) * -9999.
    
    for i in np.r_[4:nprof-4]:
        for j in np.r_[0:nlay]:
            
            if top[i,j] < 0:
                continue
            
            # look forward
            forward1, forward2, forward3 = False, False, False
            
            found1 = recouvrement(base[i,j], top[i,j], base[i+1,:], top[i+1,:])
            forward1 = (True if found1.sum() > 0 else forward1)
            
            for ilay1, f1 in enumerate(found1):
                if f1==0: continue

                found2 = recouvrement(base[i+1,ilay1], top[i+1,ilay1], base[i+2,:], top[i+2,:])
                forward2 = (True if found2.sum() > 0 else forward2)                
                
                for ilay2, f2 in enumerate(found2):
                    if f2==0: continue
                        
                    found3 = recouvrement(base[i+2,ilay2], top[i+2,ilay2], base[i+3,:], top[i+3,:])
                    forward3 = (True if found3.sum() > 0 else forward3)
                            
            backward1, backward2, backward3 = False, False, False
            
            found1 = recouvrement(base[i,j], top[i,j], base[i-1,:], top[i-1,:])
            backward1 = (True if found1.sum() > 0 else backward1)
            
            for ilay1, f1 in enumerate(found1):
                if f1==0: continue
                
                found2 = recouvrement(base[i-1,ilay1], top[i-1,ilay1], base[i-2,:], top[i-2,:])
                backward2 = (True if found2.sum() > 0 else backward2)
                for ilay2, f2 in enumerate(found2):
                    if f2==0: continue
                
                    found3 = recouvrement(base[i-2,ilay2], top[i-2,ilay2], base[i-3,:], top[i-3,:])
                    backward3 = (True if found3.sum() > 0 else backward3)
    
            if forward3 or backward3 or (backward2 and forward1) or (backward1 and forward2):
                # on a trouve une couche longue de 4 profils
                base2[i,j] = base[i,j]
                top2[i,j] = top[i,j]
                
    return base2, top2
    
def clouds_remove_1km_above_tropopause(base, top, tropoz):
    nprof = base.shape[0]
    for iprof in np.r_[0:nprof]:
        for ilayer in np.r_[0:20]:
            if base[iprof,ilayer] > (tropoz[iprof] + 1):
                base[iprof,ilayer] = -9999.
                top[iprof,ilayer] = -9999.
                
    return base, top
    
def atb_remove_below_ground(atb, alt, groundlev):
    
    nprof = atb.shape[0]
    for iprof in np.r_[0:nprof]:
        idx = alt <= groundlev[iprof]
        atb[iprof,idx] = -9999.
        
    return atb
    
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
