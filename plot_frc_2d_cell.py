import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import os
from util import *
import dxchange
import numpy as np


params_2d_cell = {'grid_delta': np.load('cell/ptychography/phantom/grid_delta.npy'),
                  'obj':[],
                  'ref':[],
                  'n_ph_tx':'4e4',
                  'save_path': 'cell/ptychography/comparison',
                  'fig_ax':[],
                  'radius_ls':[],
                  'n_ph_intersection_ls':[],
                  'radius_intersection_ls':[],
                  'T_half_bit_ls':[],
                  'show_plot_title': True,
                  'encoding_mode': 3,
                  'plot_T_half_th': True,
                  'save_mask': False,
                  }
params = params_2d_cell

print('dimension of the sample = ' +', '.join(map(str, params['grid_delta'].shape)))
n_sample_pixel = np.count_nonzero(params['grid_delta']> 1e-10)
print('n_sample_pixel = %d' %n_sample_pixel)
print('finite support area ratio in sample = %.3f' %(n_sample_pixel/(params['grid_delta'].shape[0]*params['grid_delta'].shape[1])))

if params['encoding_mode'] == 0: encoding_mode = 'normal'
if params['encoding_mode'] == 1: encoding_mode = 'comp'
if params['encoding_mode'] == 2: encoding_mode = 'dss_comp'
if params['encoding_mode'] == 3: encoding_mode = 'mix_comp'

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
#spec = gridspec.GridSpec(1, 2, width_ratios=[7, 1])
#fig = plt.figure(figsize=(8, 5))
spec = gridspec.GridSpec(1, 1)
fig = plt.figure(figsize=(6, 4))

params['fig_ax'] = fig.add_subplot(spec[0, 0])

path = os.path.dirname(params['save_path'])
n_ls = ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8']
#n_ls = ['1e4', '4e4']
n_ph_intersection_ls = []
radius_intersection_ls = []
for n_ph_tx in n_ls:
    params['n_ph_tx'] = n_ph_tx

    if encoding_mode == 'normal':
        obj_dir = os.path.join(path, 'n' + n_ph_tx)
        ref_dir = os.path.join(path, 'n' + n_ph_tx + '_ref')
    else:
        obj_dir = os.path.join(path, encoding_mode + '_n' + n_ph_tx)
        ref_dir = os.path.join(path, encoding_mode + '_n' + n_ph_tx + '_ref')

    params['obj'] = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
    params['obj'] = params['obj'][:, :, 0]
    params['ref'] = dxchange.read_tiff(os.path.join(ref_dir, 'delta_ds_1.tiff'))
    params['ref'] = params['ref'][:, :, 0]
    if params['show_plot_title']: Plot_title = encoding_mode
    else: Plot_title = None

    n_ph_intersection, radius_intersection, params['radius_ls'], params['T_half_bit_ls'] = fourier_ring_correlation_v2(**params)

    if n_ph_intersection != None:
        params['n_ph_intersection_ls'].append(n_ph_intersection)
        params['radius_intersection_ls'].append(radius_intersection)
    else:
        pass
    
if params['plot_T_half_th']:
    half_bit_threshold(params['fig_ax'], params['radius_ls'], params['T_half_bit_ls'])

#params['fig_ax'].legend(loc=3, bbox_to_anchor=(1.0, 0.0, 0.5, 0.5), fontsize=12, ncol=1, title='photon number')
plt.savefig(os.path.join(params['save_path'], 'frc_'+str(params['encoding_mode'])+'.pdf'), format='pdf')

fig.clear()
plt.close(fig)


np.savez(os.path.join(params['save_path'], 'frc_'+ str(params['encoding_mode'])+'_intersection'), np.array(params['n_ph_intersection_ls']), np.array(params['radius_intersection_ls']/params['radius_ls'][-1]))

fig = plt.figure(figsize=(8, 5))
fig_ax = fig.add_subplot(spec[0,0])
fig_ax.plot(params['n_ph_intersection_ls'], params['radius_intersection_ls']/params['radius_ls'][-1], '-bs', markerfacecolor='none', markeredgecolor='blue', label = encoding_mode)
# fig_ax.plot(n_ph_intersection_ls, radius_intersection_ls.astype(float) / radius_max, '-bs', markerfacecolor='none', markeredgecolor='blue', label = 'normal')
# fig_ax.plot(n_ph_intersection_ls, radius_intersection_ls.astype(float) / radius_max, '-ro', markerfacecolor='none', markeredgecolor='red', label = 'comp')
# fig_ax.plot(n_ph_intersection_ls, radius_intersection_ls.astype(float) / radius_max, '-gx', markerfacecolor='none', markeredgecolor='green', label = 'dss')
# fig_ax.plot(n_ph_intersection_ls, radius_intersection_ls.astype(float) / radius_max, '-kD', markerfacecolor='none', markeredgecolor='black', label = 'mix')
fig_ax.set_xlabel('Fluence (incident photons/pixel)')
fig_ax.set_ylabel('FRC/half-bit crossing fraction')
fig_ax.set_ylim(0,1.1)
fig_ax.set_xscale('log')
fig_ax.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5), ncol=1, title = 'data type')

plt.savefig(os.path.join(params['save_path'], 'frc_'+ str(params['encoding_mode'])+'_intersection.pdf'), format='pdf')
fig.clear()
plt.close(fig)