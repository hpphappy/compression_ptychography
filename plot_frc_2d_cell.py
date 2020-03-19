import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import os
from util import *
import dxchange
import numpy as np

path = 'cell/ptychography/'
fname = 'data_cell_phase.h5'
save_path = 'cell/ptychography/frc'
if not os.path.exists(save_path):
    os.makedirs(save_path)

grid_delta = np.load('cell/ptychography/phantom/grid_delta.npy')
print('dimension of the sample = ' +', '.join(map(str,grid_delta.shape)))

n_sample_pixel = np.count_nonzero(grid_delta > 1e-10)
print('n_sample_pixel = %d' %n_sample_pixel)
print('finite support area ratio in sample = %.3f' %(n_sample_pixel/(grid_delta.shape[0]*grid_delta.shape[1])))


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

n_ls = ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8']
#n_ls = ['1e4', '4e4']
step_size = 1
spec = gridspec.GridSpec(3, 2, width_ratios=[7, 1])
fig = plt.figure(figsize=(8,15))
n_ph_ls = []
fig_ax1 = fig.add_subplot(spec[0,0])
idx_intersection_normal_ls = []
n_ph_crossing_normal_ls = []
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

    radius_max = int(min(obj.shape) / 2)
    radius_ls = np.arange(1, radius_max, step_size)
    np.save(os.path.join(save_path, 'radii.npy'), radius_ls)
    f_obj = np.fft.fftshift(np.fft.fft2(obj))
    f_ref = np.fft.fftshift(np.fft.fft2(ref))
    f_obj_2 = np.abs(f_obj) ** 2
    f_ref_2 = np.abs(f_ref) ** 2
    f_prod = f_obj * np.conjugate(f_ref)

    frc_ls = []
    T_half_bit_ls = []
    for rad in radius_ls:
        print(rad)
        mask = generate_ring(obj.shape, rad, anti_aliasing=2)
        frc = abs(np.sum(f_prod * mask))
        frc /= np.sqrt(np.sum(f_obj_2 * mask) * np.sum(f_ref_2 * mask))
        frc_ls.append(frc)
        nr = np.sqrt(np.count_nonzero(mask))
        T_half_bit = (0.2071 + 1.9102 / nr) / (1.2071 + 0.97102 / nr)
        T_half_bit_ls.append(T_half_bit)
    np.save(os.path.join(save_path, 'frc' + n_ph_tx + '.npy'), frc_ls)
    fig_ax1.plot(radius_ls.astype(float) / radius_ls[-1], frc_ls, label = n_ph_ls[i])
    idx_intersection_normal = np.argwhere(np.diff(np.sign(np.array(T_half_bit_ls) - np.array(frc_ls)))).flatten()
    if len(idx_intersection_normal) != 0:
        idx_intersection_normal_ls.append(idx_intersection_normal[0]+1)
        n_ph_crossing_normal_ls.append(n_ph)
    else:
        pass




Fontsize = 12
fig_ax1.set_title('normal')
fig_ax1.set_xlabel('Spatial frequency (1 / Nyquist)', fontsize=Fontsize)
fig_ax1.set_ylabel('FRC', fontsize=Fontsize)
fig_ax1.plot(radius_ls.astype(float) / radius_ls[-1], T_half_bit_ls, 'k--',label = '1/2 bit threshold')
fig_ax1.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5),fontsize=Fontsize,ncol=1, title = 'photon number')

n_ph_ls = []
fig_ax2 = fig.add_subplot(spec[1,0])
idx_intersection_compressed_ls = []
n_ph_crossing_compressed_ls = []
for i, n_ph_tx in enumerate(n_ls):
    n_ph = float(n_ph_tx)/n_sample_pixel
    n_ph = np.round(n_ph,1)
    n_ph_ls.append(n_ph)
    obj_dir = os.path.join(path, 'comp_n' + n_ph_tx)
    ref_dir = os.path.join(path, 'comp_n' + n_ph_tx + '_ref')
    obj = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
    obj = obj[:,:,0]
    ref = dxchange.read_tiff(os.path.join(ref_dir, 'delta_ds_1.tiff'))
    ref = ref[:,:,0]

    radius_max = int(min(obj.shape) / 2)
    radius_ls = np.arange(1, radius_max, step_size)
    np.save(os.path.join(save_path, 'comp_radii.npy'), radius_ls)
    f_obj = np.fft.fftshift(np.fft.fft2(obj))
    f_ref = np.fft.fftshift(np.fft.fft2(ref))
    f_obj_2 = np.abs(f_obj) ** 2
    f_ref_2 = np.abs(f_ref) ** 2
    f_prod = f_obj * np.conjugate(f_ref)

    frc_ls = []
    T_half_bit_ls = []
    for rad in radius_ls:
        print(rad)
        mask = generate_ring(obj.shape, rad, anti_aliasing=2)
        frc = abs(np.sum(f_prod * mask))
        frc /= np.sqrt(np.sum(f_obj_2 * mask) * np.sum(f_ref_2 * mask))
        frc_ls.append(frc)

        nr = np.sqrt(np.count_nonzero(mask))
        T_half_bit = (0.2071 + 1.9102 / nr) / (1.2071 + 0.97102 / nr)
        T_half_bit_ls.append(T_half_bit)
    np.save(os.path.join(save_path, 'comp_frc' + n_ph_tx + '.npy'), frc_ls)
    fig_ax2.plot(radius_ls.astype(float) / radius_ls[-1], frc_ls, label = n_ph_ls[i])
    idx_intersection_compressed = np.argwhere(np.diff(np.sign(np.array(T_half_bit_ls) - np.array(frc_ls)))).flatten()
    if len(idx_intersection_compressed) != 0:
        idx_intersection_compressed_ls.append(idx_intersection_compressed[0]+1)
        n_ph_crossing_compressed_ls.append(n_ph)
    else:
        pass


Fontsize = 12
fig_ax2.set_title('compressed data_double step size')
fig_ax2.set_xlabel('Spatial frequency (1 / Nyquist)', fontsize=Fontsize)
fig_ax2.set_ylabel('FRC', fontsize=Fontsize)
fig_ax2.plot(radius_ls.astype(float) / radius_ls[-1], T_half_bit_ls, 'k--',label = '1/2 bit threshold')
fig_ax2.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5),fontsize=Fontsize,ncol=1, title = 'photon number')

fig_ax3 = fig.add_subplot(spec[2,0])
fig_ax3.set_xlabel('Fluence (incident photons/pixel)', fontsize=Fontsize)
fig_ax3.set_ylabel('FRC/half-bit crossing fraction', fontsize=Fontsize)
fig_ax3.plot(n_ph_crossing_normal_ls, radius_ls[idx_intersection_normal_ls].astype(float) / radius_ls[-1], '-bs', markerfacecolor='none', markeredgecolor='blue', label = 'normal')
fig_ax3.plot(n_ph_crossing_compressed_ls, radius_ls[idx_intersection_compressed_ls].astype(float) / radius_ls[-1], '-ro', markerfacecolor='none', markeredgecolor='red', label = 'compressed data')
fig_ax3.set_ylim(0,1.2)
fig_ax3.set_xscale('log')
fig_ax3.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5),fontsize=Fontsize, ncol=1, title = 'data type')

plt.savefig(os.path.join(save_path, 'frc.pdf'), format='pdf')
fig.clear()
plt.close(fig)