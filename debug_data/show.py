#!/usr/bin/env python
#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

npz = np.load('data_step1_avg.npz')

npz = np.load('data_step2_snr.npz')
lat = np.squeeze(npz['lat'])
alt = np.squeeze(npz['alt'])
atb = npz['atb']

idx = (lat < -20) & (lat > -40)
print 'Showing %d profiles' % idx.sum()

npz = np.load('data_step2.4_atbpart.npz')
atbpart = npz['atbpart']

npz = np.load('data_step2.7_cmask.npz')
cmask = npz['cmask']

npz = np.load('data_step3_cmask.npz')
clab = npz['cloud_labeled_mask']
clab = np.ma.masked_where(clab==0, clab)

plt.figure(figsize=[25,14])

plt.subplot(4,1,1)
plt.pcolormesh(lat[idx], alt, atb[idx,:].T)
plt.clim(0, 1e-3)
plt.ylim(0, 25)

plt.subplot(4,1,2)
plt.pcolormesh(lat[idx], alt, atbpart[idx,:].T)
plt.clim(0, 1e-3)
plt.ylim(0, 25)

plt.subplot(4,1,3)
plt.pcolormesh(lat[idx], alt, cmask[idx,:].T)
plt.ylim(0, 25)

plt.subplot(4,1,4)
plt.pcolormesh(lat[idx], alt, clab[idx,:].T)
plt.ylim(0, 25)


plt.show()