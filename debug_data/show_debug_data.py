#!/usr/bin/env python
#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

idx = np.r_[0:4000]

npz = np.load('data_step2_snr.npz')
lat = np.squeeze(npz['lat'][idx])
alt = np.squeeze(npz['alt'])
atb = npz['atb'][idx,:]

npz = np.load('data_step3_cmask.npz')
clab = npz['cloud_labeled_mask'][idx,:]

plt.figure(figsize=[14,10])

plt.subplot(2,1,1)
plt.pcolormesh(lat, alt, atb.T)
plt.clim(0, 1e-3)
plt.colorbar()

plt.subplot(2,1,2)
plt.pcolormesh(lat, alt, clab.T)
plt.colorbar()

plt.show()

