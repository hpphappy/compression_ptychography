import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import dxchange
from tqdm import trange
import time

np.random.seed(int(time.time()))

src_fname = 'cell/ptychography/data_cell_phase.h5'
grid_delta = np.load('cell/ptychography/phantom/grid_delta.npy')
n_sample_pixel = np.count_nonzero(grid_delta > 1e-10)
print(n_sample_pixel)

ADC_12_bit_low = 1023
ADC_12_bit_high = 4095
lows_pn = np.array([0, 16, 64, 304, 1024, 4864]) # photon number lower bounds
ups_pn = np.array([15, 63, 303, 1023, 4863, 16383]) # photon number upper bounds
lows_rv = np.array([0,20,32,80,128,320]) #reported value lower bounds
ups_rv = np.array([15,31,63,127,255,511]) #reported value lower bounds

o = h5py.File(src_fname, 'r')['exchange/data']

def in_which_gain_range(pn):
    result = np.ones_like(pn)
    for i in range(pn.shape[0]):
        for j in range(pn.shape[1]):
            result[i,j] = np.flatnonzero(np.logical_and(pn[i,j] >= lows_pn, pn[i,j] <= ups_pn))
    return result

def ADC_ouput_generator(pn):
    gain = np.where((gain_set%2==0),ADC_12_bit_low/(ups_pn[gain_set]-lows_pn[gain_set]),\
                    (ADC_12_bit_high-(ADC_12_bit_low+1))/(ups_pn[gain_set]-lows_pn[gain_set]))
    ADC_output = np.where((gain_set%2==0),(pn-lows_pn[gain_set])*gain,\
                          (pn-lows_pn[gain_set])*gain+ADC_12_bit_low+1)
    ADC_output = np.round(ADC_output)
    return ADC_output

def ADC_report_value_generator(ADC_output):
    gain_rv = np.where((gain_set%2==0),(ups_rv[gain_set]-lows_rv[gain_set])/(ADC_12_bit_low),\
                       (ups_rv[gain_set]-lows_rv[gain_set])/(ADC_12_bit_high-(ADC_12_bit_low+1))) 
    reported_value = np.where((gain_set%2==0), ADC_output*gain_rv+lows_rv[gain_set],\
                              (ADC_output-(ADC_12_bit_low+1))*gain_rv+lows_rv[gain_set])
    reported_value = np.round(reported_value)
    return reported_value


#for n_ph_tx in ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8', '1e9']:
for n_ph_tx in ['1e4', '4e4', '1e5', '4e5', '1e6', '1.75e6', '4e6', '1e7', '1.75e7', '4e7', '1e8', '1.75e8', '4e8']:
    for postfix in ['', '_ref']:

        n_ph = float(n_ph_tx) / n_sample_pixel
        # dest_fname = 'cone_256_foam_ptycho/data_cone_256_foam_1nm_n{}_temp.h5'.format(n_ph_tx)
        dest_fname = os.path.join(os.path.dirname(src_fname), os.path.splitext(os.path.basename(src_fname))[0] + '_comp_n{}{}.h5'.format(n_ph_tx, postfix))
        # dest_fname = 'cell/ptychography/data_cell_phase_n4e8.h5'


        is_ptycho = False
        if 'ptycho' in src_fname:
            is_ptycho = True
            
        file_new = h5py.File(dest_fname, 'w')
        grp = file_new.create_group('exchange')
        n = grp.create_dataset('data', dtype='complex64', shape=o.shape)
        snr_ls = []
        print('Dataset shape:')
        print(o.shape)

        # if is_ptycho:
        #     sigma = 6
        #     n_ph *= (n_sample_pixel / (o.shape[1] * 3.14 * sigma ** 2)) # photon per diffraction spot

        if is_ptycho:

            # total photons received by sample
            n_ex = n_ph * n_sample_pixel
            # total photons per image
            n_ex *= (float(grid_delta.size) / n_sample_pixel)
            # total photons per spot
            n_ex /= o.shape[1]
            print(o.shape[1])

            for i in trange(o.shape[0]):
                for j in range(o.shape[1]):
                    prj_o = o[i, j]
                    prj_o_inten = np.abs(prj_o) ** 2

                    spot_integral = np.sum(prj_o_inten)
                    multiplier = n_ex / spot_integral
                    # scale intensity to match expected photons per spot
                    pro_o_inten_scaled = prj_o_inten * multiplier
                    # dc_intensity = prj_o_inten[int(o.shape[-2] / 2), int(o.shape[-1] / 2)]
                    # prj_o_inten_norm = prj_o_inten / dc_intensity
                    print(n_ph)
                    prj_o_inten_noisy = np.random.poisson(pro_o_inten_scaled)
#                    prj_o_inten_noisy = prj_o_inten_noisy / multiplier
                    gain_set = in_which_gain_range(prj_o_inten_noisy)
                    ADC_output = ADC_ouput_generator(prj_o_inten_noisy)
                    value_comprs = ADC_report_value_generator(ADC_output)
                    
                    pn_each_step = np.array([1, 4, 7.5, 15, 30, 60])
                    base_pn = np.array([0, 19, 70.5, 318, 1053, 4923])
                    prj_o_inten_noisy_decomprs = (value_comprs - lows_rv[gain_set])*pn_each_step[gain_set]+base_pn[gain_set]
                    
#                    noise = prj_o_inten_noisy - prj_o_inten
#                    noise = prj_o_inten_noisy - prj_o_inten_scaled
#                    snr = np.var(prj_o_inten) / np.var(noise)
#                    snr = np.var(prj_o_inten_scaled) / np.var(noise)
#                    snr_ls.append(snr)
                    data = np.sqrt(prj_o_inten_noisy_decomprs)
                    n[i, j] = data.astype('complex64')

        else:
            print('FF')
            for i in trange(o.shape[0]):
                prj_o = o[i]
                prj_o_inten = np.abs(prj_o) ** 2
                prj_o_inten_noisy = np.random.poisson(prj_o_inten * n_ph)
                # noise = prj_o_inten_noisy - prj_o_inten
                # print(np.var(noise))
                prj_o_inten_noisy = prj_o_inten_noisy / n_ph
                noise = prj_o_inten_noisy - prj_o_inten
                snr = np.var(prj_o_inten) / np.var(noise)
                snr_ls.append(snr)
                data = np.sqrt(prj_o_inten_noisy)
                n[i] = data.astype('complex64')

        print('Average SNR is {}.'.format(np.mean(snr_ls)))

        dxchange.write_tiff(abs(n[0]), os.path.join(os.path.dirname(dest_fname), 'comp_n{}{}'.format(n_ph_tx, postfix)), dtype='float32', overwrite=True)


        # ------- based on SNR -------
        # snr = 10

        # for i in tqdm(range(o.shape[0])):
        # for i in tqdm(range(1)):
        #     prj_o = o[i]
        #     prj_o_inten = np.abs(prj_o) ** 2
        #     var_signal = np.var(prj_o_inten)
        #     var_noise = var_signal / snr

            # noise = np.random.poisson(prj_o_inten * (var_noise / np.mean(prj_o_inten)) * 1e10) / 1e5
            # noise = noise * np.sqrt(var_noise / np.var(noise))
            # noise -= np.mean(noise)
            # prj_n_inten = prj_o_inten + noise
            # prj_n = prj_o * np.sqrt(prj_n_inten / prj_o_inten)
            #
            # n[i] = prj_n
