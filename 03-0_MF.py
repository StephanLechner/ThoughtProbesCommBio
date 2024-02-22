import mne
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
from pymatreader import read_mat
import glob
import os
from datetime import datetime
from sys import exit
from scipy import signal

mne.set_log_level('WARNING')
startTime = datetime.now()

def welch_psd(data, freq_l, fs, preserve_memory=False, verbose=False):
    n_sample = data.shape[0]
    n_fft = 2 ** np.ceil(np.log2(n_sample)) if not preserve_memory else n_sample
    win_size = int(2 * 1.6 / freq_l * fs)
    overlap = int(0.9 * win_size)

    if win_size > n_sample:
        raise ValueError(f"Window size ({win_size}) is larger than the total numbeer of samples ({n_sample})")

    if verbose:
        print(f"Welch PSD with following parameters: nfft: {n_fft}, no detrend\n"
            f"Window: hanning, size {win_size}, overlap {overlap}")

    return signal.welch(data, fs=fs, nfft=n_fft, detrend=None,
                        window="hamming", nperseg=win_size, noverlap=overlap)

def sub_mf(EEG,sf,low_pass,skip):
    epoch_mf = np.zeros((20,63))
    move_down=0

    #Get MF for each electrode
    for epoch in range(20):
        if epoch in skip:
            epoch_mf[epoch,:] = np.nan
            epoch+=1
            move_down += 1
            continue
        electrodes_mf = []			#this collects single MF values for each electrode
        for chan in range(EEG.shape[1]):
            ts=EEG[epoch-move_down][chan]
            freq,psd = welch_psd(ts,low_pass,500)
            cum_sum = np.cumsum(psd)
            single_elec_mf = freq[np.where(cum_sum>np.max(cum_sum)/2)[0][0]]   
            # print(epoch,'\t',chan,'\t\t',single_elec_mf)
            electrodes_mf.append(single_elec_mf)	#adds value of electrode to subject container
        epoch_mf[epoch,:]=electrodes_mf
        electrodes_mf_mean = np.mean(electrodes_mf)
    return epoch_mf,electrodes_mf, electrodes_mf_mean

outs = np.load('./data/data_generated/00_out_index5-30.npy',allow_pickle=True)

freqs_low_pass = [40]
for low_pass in freqs_low_pass:

    f_names = []
    ndar = []
    path = '../data/data_raw/resting_state_thought_probe'
    for filename in sorted(glob.glob(os.path.join(path, 'Sub*'))):
        f_names.append(filename)

    list_epoch_lengths = [-3,-4,-5,-10,-15,-20,-25,-30]

    mf  = np.zeros((35,len(list_epoch_lengths),20,63),dtype=object)
    out_index=np.zeros((35,1),dtype=object)

    for subject in range(0,len(f_names)):

        filename = f_names[subject]
        dat = read_mat(filename)['EEG_rest']

        ''' delete all non 9 and 32 makers'''
        myIndices = [i for i, x in enumerate(dat['event']['type']) if x in ['S 32','S  9']]
        dat['event']['type'] = [dat['event']['type'][i] for i in myIndices]
        dat['event']['latency'] = [dat['event']['latency'][i] for i in myIndices]

        if subject == 20:   #for some reason sub 20 has some more event markers at the end of the trial.
            dat['event']['type'] = dat['event']['type'][:40]
            dat['event']['latency'] = dat['event']['latency'][:40]

        for i in range(len(dat['event']['type'])):
            if dat['event']['type'][i] == 'S 32':
                dat['event']['type'][i]= 32
            elif dat['event']['type'][i] == 'S  9':
                dat['event']['type'][i]= 9
        markers=dat['event']['type']

        events_interpretation       = np.empty((0,4), int)
        events                      = np.empty((0,3), int)
        for i in range(len(dat['event']['type'])):
            if dat['event']['type'][i] == 9:
                single_interpretation = np.array((int(dat['event']['latency'][i]/500),0,int(dat['event']['type'][i]),i))
                single = np.array((int(dat['event']['latency'][i]),0,int(dat['event']['type'][i])))
                events_interpretation = np.append(events_interpretation,[single_interpretation],axis=0)
                events = np.append(events,[single],axis=0)

        ch_names = dat['chanlocs']['labels']
        info = mne.create_info( ch_names=ch_names,
                                sfreq    = dat['srate'],
                                ch_types = ['eeg']*int(dat['nbchan']))
        raw = mne.io.RawArray(dat['data'],info)
        raw.filter(0.5, low_pass, verbose=False)

        event_ids = {'probe':9}
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
        baseline = (None, 0.0)

        mf_sub  = np.zeros((len(list_epoch_lengths),20,63),dtype=object)
        for i,epoch_len in enumerate(list_epoch_lengths):
            
            tmin ,tmax = epoch_len,0
            epochs =  mne.Epochs(raw, events=events, event_id=event_ids, tmin=tmin, tmax=tmax , baseline=baseline, picks=picks)

            EEG = epochs.get_data()
            skip=outs[i][subject][0]

            mf_return = sub_mf(EEG,500,low_pass,skip=skip)
            mf[subject,i] = mf_return[0]
            mf_sub[i] = mf_return[0]
 
        np.save('./data/data_generated/mf_subs/sub_{}_mf_0.5_'.format(subject)+str(low_pass),mf_sub)


