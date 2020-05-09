import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


data_0 = np.load('cell/ptychography/comparison/frc_0_intersection.npz')
data_1 = np.load('cell/ptychography/comparison/frc_1_intersection.npz')
data_2 = np.load('cell/ptychography/comparison/frc_2_intersection.npz')
data_3 = np.load('cell/ptychography/comparison/frc_3_intersection.npz')

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
spec = gridspec.GridSpec(1, 2, width_ratios=[7, 0])

fig = plt.figure(figsize=(8, 5))
fig_ax = fig.add_subplot(spec[0,0])

fig_ax.plot(data_0['arr_0'], data_0['arr_1'], '-bs', markerfacecolor='none', markeredgecolor='blue', label = 'without encoding')
fig_ax.plot(data_1['arr_0'], data_1['arr_1'], '-ro', markerfacecolor='none', markeredgecolor='red', label = 'encoding scheme (a)')
fig_ax.plot(data_3['arr_0'], data_3['arr_1'], '-kD', markerfacecolor='none', markeredgecolor='black', label = 'encoding scheme (b)')
fig_ax.plot(data_2['arr_0'], data_2['arr_1'], '-gx', markerfacecolor='none', markeredgecolor='green', label = 'encoding scheme (c)')

fig_ax.set_xlabel('Fluence (incident photons/pixel)')
fig_ax.set_ylabel('FRC/half-bit crossing fraction')
fig_ax.set_ylim(0,1.1)
fig_ax.set_xscale('log')
fig_ax.legend(loc=2, ncol=1, title = 'data type')

plt.savefig(os.path.join('cell/ptychography/comparison', 'frc.pdf'), format='pdf')
#plt.savefig(os.path.join('cell/ptychography/comparison', 'frc.png'))
fig.clear()
plt.close(fig)
