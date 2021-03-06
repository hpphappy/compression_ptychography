import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
matplotlib.use('pdf')
import h5py
import os
from util import *
import dxchange
import numpy as np

path = 'cell/ptychography/'

save_path = os.path.join(path, 'comparison')
if not os.path.exists(save_path):
    os.makedirs(save_path)

grid_delta = np.load('cell/ptychography/phantom/grid_delta.npy')
print('dimension of the sample = ' +', '.join(map(str,grid_delta.shape)))

n_sample_pixel = np.count_nonzero(grid_delta > 1e-10)
finite_support = (grid_delta > 1e-10).astype('float')
print('n_sample_pixel = %d' %n_sample_pixel)
print('finite support area ratio in sample = %.3f' %(n_sample_pixel/(grid_delta.shape[0]*grid_delta.shape[1])))


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

n_ls = ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8']
step_size = 1
spec = gridspec.GridSpec(1, 2, width_ratios=[7, 0])
fig = plt.figure(figsize=(8,5))
n_ph_ls = []
snr_normal_ls = []
fig_ax1 = fig.add_subplot(spec[0,0])
for i, n_ph_tx in enumerate(n_ls):
    n_ph = float(n_ph_tx)/n_sample_pixel
    n_ph = np.round(n_ph,1)
    n_ph_ls.append(n_ph)
    obj_dir = os.path.join(path, 'n' + n_ph_tx)
    ref_dir = os.path.join(path, 'n' + n_ph_tx + '_ref')
    obj = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
    obj = obj[:,:,0]
    ref = dxchange.read_tiff(os.path.join(ref_dir, 'delta_ds_1.tiff'))
    ref = ref[:,:,0]

    r_numerator = np.average(finite_support*(obj-np.average(obj))*np.conjugate(ref-np.average(ref)))
    r_denominator = np.sqrt(np.average(finite_support*(obj-np.average(obj))**2)*np.average(finite_support*(ref-np.average(ref))**2))
    r = r_numerator/r_denominator
    snr = np.sqrt(r/(1-r))
    snr_normal_ls.append(snr)

fig_ax1.plot(n_ph_ls, snr_normal_ls, '-bs', markerfacecolor='none', markeredgecolor='blue', label = 'without encoding')


snr_comp_ls = []
for i, n_ph_tx in enumerate(n_ls):
    obj_comp_dir = os.path.join(path, 'comp_n' + n_ph_tx)
    ref_comp_dir = os.path.join(path, 'comp_n' + n_ph_tx + '_ref')
    obj_comp = dxchange.read_tiff(os.path.join(obj_comp_dir, 'delta_ds_1.tiff'))
    obj_comp = obj_comp[:,:,0]
    ref_comp = dxchange.read_tiff(os.path.join(ref_comp_dir, 'delta_ds_1.tiff'))
    ref_comp = ref_comp[:,:,0]

    r_numerator = np.average(finite_support*(obj_comp-np.average(obj_comp))*np.conjugate(ref_comp-np.average(ref_comp)))
    r_denominator = np.sqrt(np.average(finite_support*(obj_comp-np.average(obj_comp))**2)*np.average(finite_support*(ref_comp-np.average(ref_comp))**2))
    r = r_numerator/r_denominator
    snr_comp = np.sqrt(r/(1-r))
    snr_comp_ls.append(snr_comp)

fig_ax1.plot(n_ph_ls, snr_comp_ls, '-ro', markerfacecolor='none', markeredgecolor='red', label = 'encoding scheme (a)')

snr_mix_comp_ls = []
for i, n_ph_tx in enumerate(n_ls):
    obj_comp_dir = os.path.join(path, 'mix_comp_n' + n_ph_tx)
    ref_comp_dir = os.path.join(path, 'mix_comp_n' + n_ph_tx + '_ref')
    obj_comp = dxchange.read_tiff(os.path.join(obj_comp_dir, 'delta_ds_1.tiff'))
    obj_comp = obj_comp[:,:,0]
    ref_comp = dxchange.read_tiff(os.path.join(ref_comp_dir, 'delta_ds_1.tiff'))
    ref_comp = ref_comp[:,:,0]

    r_numerator = np.average(finite_support*(obj_comp-np.average(obj_comp))*np.conjugate(ref_comp-np.average(ref_comp)))
    r_denominator = np.sqrt(np.average(finite_support*(obj_comp-np.average(obj_comp))**2)*np.average(finite_support*(ref_comp-np.average(ref_comp))**2))
    r = r_numerator/r_denominator
    snr_mix_comp = np.sqrt(r/(1-r))
    snr_mix_comp_ls.append(snr_mix_comp)

fig_ax1.plot(n_ph_ls, snr_mix_comp_ls, '-kD', markerfacecolor='none', markeredgecolor='black', label = 'encoding scheme (b)')


snr_dss_comp_ls = []
for i, n_ph_tx in enumerate(n_ls):
    obj_comp_dir = os.path.join(path, 'dss_comp_n' + n_ph_tx)
    ref_comp_dir = os.path.join(path, 'dss_comp_n' + n_ph_tx + '_ref')
    obj_comp = dxchange.read_tiff(os.path.join(obj_comp_dir, 'delta_ds_1.tiff'))
    obj_comp = obj_comp[:,:,0]
    ref_comp = dxchange.read_tiff(os.path.join(ref_comp_dir, 'delta_ds_1.tiff'))
    ref_comp = ref_comp[:,:,0]

    r_numerator = np.average(finite_support*(obj_comp-np.average(obj_comp))*np.conjugate(ref_comp-np.average(ref_comp)))
    r_denominator = np.sqrt(np.average(finite_support*(obj_comp-np.average(obj_comp))**2)*np.average(finite_support*(ref_comp-np.average(ref_comp))**2))
    r = r_numerator/r_denominator
    snr_dss_comp = np.sqrt(r/(1-r))
    snr_dss_comp_ls.append(snr_dss_comp)

fig_ax1.plot(n_ph_ls, snr_dss_comp_ls, '-gx', markerfacecolor='none', markeredgecolor='green', label = 'encoding scheme (c)')



Fontsize = 12
fig_ax1.set_xlabel('Fluence (incident photons/pixel)', fontsize=Fontsize)
fig_ax1.set_ylabel('SNR', fontsize=Fontsize)
fig_ax1.legend(loc=4, fontsize=Fontsize, ncol=1, title = 'data type')
fig_ax1.set_xscale('log')
fig_ax1.set_yscale('log')

plt.savefig(os.path.join(save_path, 'snr.pdf'), format='pdf')
#plt.savefig(os.path.join(save_path, 'snr.png'))
fig.clear()
plt.close(fig)