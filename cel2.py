#!/usr/bin/env python
# encoding: utf-8

'''
cel2.py

Created by V. Noel on 2011-05-17

'''

import numpy as np
from datetime import datetime
import os
try:
    import netCDF4
except:
    import netCDF3 as netCDF4


cel2_version_major = 1
cel2_version_minor = 0
netcdf_format = 'NETCDF4_CLASSIC'
netcdf_extension = 'nc4'

# this can be overridden by cel2_process_period.py
outpath = './test_output/'
outbase = 'CEL2-V%1d-%02d.' % (cel2_version_major, cel2_version_minor)

debug = False
debug_file = None

# maximum number of layers per profile
nl = 30

class cel2_data(object):
    
    def __init__(self, year, month, day, orbit, orig_l1_file):
        self.date = datetime(year, month, day)
        self.orbit = orbit
        self.orig_l1_file = orig_l1_file
        self.filepath = outpath + '%04d/%04d_%02d_%02d' % (year, year, month, day)
        self.filename = outbase + '%04d-%02d-%02d%s.%s' % (year, month, day, self.orbit, netcdf_extension)
        self.nl = nl
    
        
    def _layer_data(self, nprof):
        return np.ones([nprof, self.nl]) * -9999.
    
    
    def init_data(self, nprof):
        self.nprof = nprof
        self.time = np.ones([nprof]) * -9999.
        self.longitude = np.ones([nprof]) * -9999.
        self.latitude = np.ones([nprof]) * -9999.
        self.layer_top_altitude = self._layer_data(nprof)
        self.layer_base_altitude = self._layer_data(nprof)
        self.midlayer_temperature = self._layer_data(nprof)
        self.integrated_attenuated_backscatter_532 = self._layer_data(nprof)
        self.feature_optical_depth = self._layer_data(nprof)
        self.integrated_particulate_depolarization_ratio = self._layer_data(nprof)
        self.integrated_volume_depolarization_ratio = self._layer_data(nprof)
        self.integrated_particulate_color_ratio = self._layer_data(nprof)
        self.integrated_volume_color_ratio = self._layer_data(nprof)
        self.particulate_parallel_backscatter = self._layer_data(nprof)
        self.particulate_perpendicular_backscatter = self._layer_data(nprof)
        self.cloud_id = self._layer_data(nprof)
        self.opacity_flag = self._layer_data(nprof)
        
        
    def set_nlayers_max(self, nlmax):
        self.layer_top_altitude = self.layer_top_altitude[:,:nlmax]
        self.layer_base_altitude = self.layer_base_altitude[:,:nlmax]
        self.midlayer_temperature = self.midlayer_temperature[:,:nlmax]
        self.integrated_attenuated_backscatter_532 = self.integrated_attenuated_backscatter_532[:,:nlmax]
        self.feature_optical_depth = self.feature_optical_depth[:,:nlmax]
        self.integrated_particulate_depolarization_ratio = integrated_particulate_depolarization_ratio[:,:nlmax]
        self.integrated_volume_depolarization_ratio = self.integrated_volume_depolarization_ratio[:,:nlmax]
        self.integrated_particulate_color_ratio = self.integrated_particulate_color_ratio[:,:nlmax]
        self.integrated_volume_color_ratio = self.integrated_volume_color_ratio[:,:nlmax]
        self.particulate_parallel_backscatter = self.particulate_parallel_backscatter[:,:nlmax]
        self.particulate_perpendicular_backscatter = self.particulate_perpendicular_backscatter[:,:nlmax]
        self.cloud_id = self.cloud_id[:,:nlmax]
        self.opacity_flag = self.opacity_flag[:,:nlmax]
        
        
    def invalid_data_based_on(self, mask):
        idx = (mask == -9999)
        self.midlayer_temperature[idx] = -9999.
        self.integrated_attenuated_backscatter_532[idx] = -9999.
        self.feature_optical_depth[idx] = -9999.
        self.integrated_particulate_depolarization_ratio[idx] = -9999.
        self.integrated_volume_depolarization_ratio[idx] = -9999.
        self.integrated_particulate_color_ratio[idx] = -9999.
        self.integrated_volume_color_ratio[idx] = -9999.
        self.particulate_parallel_backscatter[idx] = -9999.
        self.particulate_perpendicular_backscatter[idx] = -9999.
        
        
    def set_time(self, time):
        self.time[:] = time[:]
        
        
    def set_coords(self, lon, lat):
        self.longitude[:] = np.squeeze(lon[:])
        self.latitude[:] = np.squeeze(lat[:])
    
    
    def set_layers(self, base, top):
        self.layer_top_altitude[:,:] = top[:,:]
        self.layer_base_altitude[:,:] = base[:,:]
    
        
    def set_iatb(self, iatb):
        self.integrated_attenuated_backscatter_532[:,:] = iatb[:,:]
    
        
    def set_temperature(self, temp):
        self.midlayer_temperature[:,:] = temp[:,:]
    
        
    def set_volume_depolarization(self, vdp):
        self.integrated_volume_depolarization_ratio[:,:] = vdp[:,:]
    
        
    def set_particulate_depolarization(self, pdp):
        self.integrated_particulate_depolarization_ratio[:,:] = pdp[:,:]
    
        
    def set_particulate_parallel_backscatter(self, para):
        self.particulate_parallel_backscatter[:,:] = para


    def set_particulate_perpendicular_backscatter(self, perp):
        self.particulate_perpendicular_backscatter[:,:] = perp

    
    def set_volume_color_ratio(self, vcr):
        self.integrated_volume_color_ratio[:,:] = vcr[:,:]

    
    def set_particulate_color_ratio(self, pcr):
        self.integrated_particulate_color_ratio[:,:] = pcr[:,:]

        
    def set_od(self, od):
        self.feature_optical_depth[:,:] = od[:,:]

        
    def set_opacity_flag(self, opacity):
        self.opacity_flag[:,:] = opacity[:,:]

        
    def set_horizontal_extension(self, hext):
        self.horizontal_extension = hext

        
    def set_cloud_id(self, cloud_id):
        self.cloud_id = cloud_id        


    def save(self):
                
        print 'Saving '+self.filepath+'/'+self.filename
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
        nc = netCDF4.Dataset(self.filepath + '/' + self.filename, 'w', format=netcdf_format)
        
        # time is an unlimited dimension
        nc.createDimension('time', None)
        nc.createDimension('nmaxlayers', nl)
        nc.createDimension('nclouds', np.max(self.cloud_id) + 1)
        
        nc.createVariable('Profile_Time', 'f8', ('time',))
        nc.createVariable('longitude', 'f8', ('time',))
        nc.createVariable('latitude', 'f8', ('time',))
        nc.createVariable('layer_top_altitude', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('layer_base_altitude', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('layer_temperature', 'f4', ('time', 'nmaxlayers'), zlib=True)
        # nc.createVariable('integrated_attenuated_backscatter_532', 'f4', ('time', 'nmaxlayers'), zlib=True)
        # nc.createVariable('integrated_particulate_parallel_backscatter_532', 'f4', ('time', 'nmaxlayers'), zlib=True)
        # nc.createVariable('integrated_particulate_perpendicular_backscatter_532', 'f4', ('time', 'nmaxlayers'), zlib=True)
        # nc.createVariable('integrated_particulate_depolarization_ratio', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('integrated_volume_depolarization_ratio', 'f4', ('time', 'nmaxlayers'), zlib=True)
        # nc.createVariable('integrated_particulate_color_ratio', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('integrated_volume_color_ratio', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('layer_optical_depth', 'f4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('layer_opacity_flag', 'i1', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('layer_cloud_id', 'i4', ('time', 'nmaxlayers'), zlib=True)
        nc.createVariable('cloud_horizontal_extension', 'f4', ('nclouds',), zlib=True)
        
        nc.variables['Profile_Time'][:] = self.time
        nc.variables['longitude'][:] = self.longitude
        nc.variables['latitude'][:] = self.latitude
        nc.variables['layer_top_altitude'][:,:] = self.layer_top_altitude
        nc.variables['layer_base_altitude'][:,:] = self.layer_base_altitude
        nc.variables['layer_temperature'][:,:] = self.midlayer_temperature
        # nc.variables['integrated_attenuated_backscatter_532'][:,:] = self.integrated_attenuated_backscatter_532
        # nc.variables['integrated_particulate_parallel_backscatter_532'][:,:] = self.particulate_parallel_backscatter
        # nc.variables['integrated_particulate_perpendicular_backscatter_532'][:,:] = self.particulate_perpendicular_backscatter
        # nc.variables['integrated_particulate_depolarization_ratio'][:,:] = self.integrated_particulate_depolarization_ratio
        nc.variables['integrated_volume_depolarization_ratio'][:,:] = self.integrated_volume_depolarization_ratio
        # nc.variables['integrated_particulate_color_ratio'][:,:] = self.integrated_particulate_color_ratio
        nc.variables['integrated_volume_color_ratio'][:,:] = self.integrated_volume_color_ratio
        nc.variables['layer_optical_depth'][:,:] = self.feature_optical_depth
        nc.variables['layer_opacity_flag'][:,:] = self.opacity_flag
        nc.variables['layer_cloud_id'][:,:] = self.cloud_id
        nc.variables['cloud_horizontal_extension'][:] = self.horizontal_extension
        
        nc.variables['Profile_Time'].units = "seconds...TAI"
        nc.variables['Profile_Time'].description = "Profile Time of the 1st CALIOP Level 1 profile used for averaging"
        nc.variables['longitude'].units = 'degrees East'
        nc.variables['latitude'].units = 'degrees North'
        nc.variables['layer_temperature'].units = 'degrees Celsius'
        nc.variables['layer_top_altitude'].units = 'km'
        nc.variables['layer_base_altitude'].units = 'km'
        # nc.variables['integrated_attenuated_backscatter_532'].units = 'km-1'
        # nc.variables['integrated_particulate_parallel_backscatter_532'].units = 'km-1'
        # nc.variables['integrated_particulate_perpendicular_backscatter_532'].units = 'km-1'
        
        nc.variables['cloud_horizontal_extension'].units = 'km'

        nc.variables['layer_opacity_flag'].description = '1 if no ground return can be found under the layer, 0 otherwise.\
                                                          If 1, base altitude is not reliable for this layer.'

        import time
        nc.description = 'CEL2 profile layer properties.'
        nc.history = 'Created '+time.ctime(time.time())+' from %s' % (self.orig_l1_file)
        nc.orbit = self.orbit
        nc.author = 'V. Noel, LMD/IPSL/CNRS'
        nc.version='%d.%02d' % (cel2_version_major, cel2_version_minor)
        nc.invalid_data = -9999.
        
        nc.close()

def main():
    pass

if __name__ == '__main__':
    main()
