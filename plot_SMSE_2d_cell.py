import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import os
from util import *
import dxchange
import numpy as np

path = 'cell/ptychography/'
save_path = 'cell/ptychography/smse'
if not os.path.exists(save_path):
    os.makedirs(save_path)

grid_delta = np.load('cell/ptychography/phantom/grid_delta.npy')
print('dimension of the sample = ' +', '.join(map(str,grid_delta.shape)))
grid_delta = grid_delta[:,:,0]

n_sample_pixel = np.count_nonzero(grid_delta > 1e-10)
print('n_sample_pixel = %d' %n_sample_pixel)
print('finite support area ratio in sample = %.3f' %(n_sample_pixel/(grid_delta.shape[0]*grid_delta.shape[1])))


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

n_ls = ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8']
step_size = 1
spec = gridspec.GridSpec(1, 2, width_ratios=[7, 1])
fig = plt.figure(figsize=(8,5))
n_ph_ls = []
smse_normal_ls = []
fig_ax1 = fig.add_subplot(spec[0,0])
for i, n_ph_tx in enumerate(n_ls):
    n_ph = float(n_ph_tx)/n_sample_pixel
    n_ph = np.round(n_ph,1)
    n_ph_ls.append(n_ph)
    obj_dir = os.path.join(path, 'n' + n_ph_tx)
    obj = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
    obj = obj[:,:,0]
    smse_normal = np.sum((grid_delta - obj)**2)/n_sample_pixel
    smse_normal_ls.append(smse_normal)

fig_ax1.plot(n_ph_ls, smse_normal_ls, '-bs', markerfacecolor='none', markeredgecolor='blue', label = 'normal')

smse_comp_ls = []
for i, n_ph_tx in enumerate(n_ls):
    obj_comp_dir = os.path.join(path, 'comp_n' + n_ph_tx)
    ref_comp_dir = os.path.join(path, 'comp_n' + n_ph_tx + '_ref')
    obj_comp = dxchange.read_tiff(os.path.join(obj_comp_dir, 'delta_ds_1.tiff'))
    obj_comp = obj_comp[:,:,0]
    smse_comp = np.sum((grid_delta - obj_comp)**2)/n_sample_pixel
    smse_comp_ls.append(smse_comp)

fig_ax1.plot(n_ph_ls, smse_comp_ls, '-ro', markerfacecolor='none', markeredgecolor='red', label = 'compressed')

Fontsize = 12
fig_ax1.set_xlabel('Fluence (incident photons/pixel)', fontsize=Fontsize)
fig_ax1.set_ylabel('SMSE', fontsize=Fontsize)
fig_ax1.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5),fontsize=Fontsize,ncol=1)
fig_ax1.set_xscale('log')
fig_ax1.set_yscale('log')

plt.savefig(os.path.join(save_path, 'SMSE.pdf'), format='pdf')
fig.clear()
plt.close(fig)