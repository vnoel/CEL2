#!/usr/bin/env python
#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

npz = np.load('data_step3_cmask.npz')

lat = np.squeeze(npz['lat'])
alt = np.squeeze(npz['alt'])

clab = npz['cloud_labeled_mask']
clab = np.ma.masked_where(clab==0, clab)
nclouds = np.max(clab)
print 'Number of independent clouds : ', nclouds

hext  = npz['hext']

idx_s = (lat < -20) & (lat > -55)
idx_n = (lat > 20) & (lat < 55)

plt.figure(figsize=[25,14])

plt.subplot(4,1,1)
plt.pcolormesh(lat[idx_n], alt, clab[idx_n,:].T)
plt.ylim(0, 25)

plt.subplot(4,1,3)
plt.pcolormesh(lat[idx_s], alt, clab[idx_s,:].T)
plt.ylim(0, 25)

# remove clouds with non-wanted horizontal extension

print 'Cleaning up clouds'
idx = (hext < 0.5) | (hext > 2.)
cloud_id_to_remove = np.arange(nclouds+1)[idx]
for j in np.r_[0:583]:
    idx = np.in1d(clab[:,j], cloud_id_to_remove)
    clab[idx,j] = 0
clab = np.ma.masked_equal(clab, 0)
# for i in np.arange(nclouds+1)[idx]:
#     print 'Removing cloud #', i, 'with extension ', hext[i]
#     # idx = (clab==i)
#     # clab[idx] = 0
#     clab = np.ma.masked_equal(clab, i)

plt.subplot(4,1,2)
plt.pcolormesh(lat[idx_n], alt, clab[idx_n,:].T)
plt.ylim(0, 25)

plt.subplot(4,1,4)
plt.pcolormesh(lat[idx_s], alt, clab[idx_s,:].T)
plt.ylim(0, 25)

plt.show()