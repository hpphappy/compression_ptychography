from ptychography import reconstruct_ptychography
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--save_path', default='cell/ptychography')
parser.add_argument('--output_folder', default='test') # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]


params_2d_cell = {'fname': 'data_cell_phase.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'theta_downsample': 1,
                    'n_epochs': 200,
                    'obj_size': (325, 325, 1),
                    'alpha_d': 0,
                    'alpha_b': 0,
                    'gamma': 0,
                    'probe_size': (72, 72),
                    'learning_rate': 4e-3,
                    'center': 512,
                    'energy_ev': 5000,
                    'psize_cm': 1.e-7,
                    'minibatch_size': 4488,
                    'n_batch_per_update': 1,
                    'cpu_only': True,
                    'save_path': 'cell/ptychography',
                    'multiscale_level': 1,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    # 'initial_guess': [np.zeros([325, 325, 1]) + 0.032, np.zeros([325, 325, 1])],
                    'initial_guess': None,
                    'n_dp_batch': 20,
                    'probe_type': 'gaussian',
                    'probe_mag_sigma': 6,
                    'probe_phase_sigma': 6,
                    'probe_phase_max': 0.5,
                    'forward_algorithm': 'fresnel',
                    'object_type': 'phase_only',
                    'probe_pos': [(y, x) for y in (np.arange(66) * 5) - 36 for x in (np.arange(68) * 5) - 36],
                    'finite_support_mask': None,
                    'free_prop_cm': 'inf',
                    'optimizer': 'adam',
                    'two_d_mode': True,
                    'shared_file_object': False,
                    'use_checkpoint': False
                    }

params = params_2d_cell

# n_ls = ['nonoise', 'n1e9', 'n1e8', 'n1e7', 'n1e6', 'n1e5', 'n1e4']
#n_ls = ['nonoise', 'comp_n1e4', 'comp_n4e4', 'comp_n1e5', 'comp_n4e5', 'comp_n1e6', 'comp_n1.75e6', 'comp_n4e6', 'comp_n1e7', 'comp_n1.75e7', 'comp_n4e7', 'comp_n1e8', 'comp_n1.75e8', 'comp_n4e8']
n_ls = ['comp_n1e4', 'comp_n4e4', 'comp_n1e5', 'comp_n4e5', 'comp_n1e6', 'comp_n1.75e6', 'comp_n4e6', 'comp_n1e7', 'comp_n1.75e7', 'comp_n4e7', 'comp_n1e8', 'comp_n1.75e8', 'comp_n4e8']
n_ls = [x + '_ref' for x in n_ls]

for n_ph in n_ls:
    if 'nonoise' in n_ph:
        params['fname'] = 'data_cell_phase.h5'
    else:
        params['fname'] = 'data_cell_phase_{}.h5'.format(n_ph)
    params['output_folder'] = n_ph
    reconstruct_ptychography(**params)
