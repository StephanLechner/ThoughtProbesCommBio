import mne
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
from pymatreader import read_mat
import glob
import os
from datetime import datetime


from statsmodels.tsa import stattools
def sub_ACW(EEG,sf,epoched,skip):
    if epoched==True:
        epoch_acw = np.zeros((20,63))
        epoch_acz = np.zeros((20,63))
        # epoch_acw=np.empty()
        move_down=0
        #Get ACW for each electrode
        for epoch in range(20):
            if epoch in skip:
                epoch_acw[epoch,:] = np.nan
                epoch_acz[epoch,:] = np.nan
                epoch+=1
                move_down += 1
                continue
            electrodes_acw = []			#this collects single acw values for each electrode
            electrodes_acz = []	
            collector = []
            for chan in range(EEG.shape[1]):
                ts=EEG[epoch-move_down][chan]
                n_lag=len(ts)
                acf = stattools.acf(ts,nlags=n_lag,qstat=False,alpha=None,fft=True)
                collector.append(acf)
                single_elec_acw = (np.argmax(acf<=0.5)/500)	#ACW value for a single electrode 
                single_elec_acz = (np.argmax(acf<=0)/500)	#ACW value for a single electrode 
                electrodes_acw.append(single_elec_acw)	#adds value of electrode to subject container
                electrodes_acz.append(single_elec_acz)	#adds value of electrode to subject container
            epoch_acw[epoch,:]=electrodes_acw
            epoch_acz[epoch,:]=electrodes_acz

            # electrodes_acw_cv
            electrodes_acw_mean = np.mean(electrodes_acw)
            electrodes_acz_mean = np.mean(electrodes_acz)
            mean_acf=np.mean(collector,axis=0)
        return epoch_acw,epoch_acz,electrodes_acw, electrodes_acz, electrodes_acw_mean,electrodes_acz_mean, mean_acf

mne.set_log_level('WARNING')
startTime = datetime.now()

outs = np.load('./data/data_generated/00out_index5-30.npy',allow_pickle=True)
freqs_low_pass =[40]

for low_pass in freqs_low_pass:
    f_names = []
    ndar = []
    path = '../data/data_raw/resting_state_thought_probe'
    for filename in sorted(glob.glob(os.path.join(path, 'Sub*'))):
        f_names.append(filename)

    list_epoch_lengths = [-3,-4,-5,-10,-15,-20,-25,-30]     #seconds before thought probes
    acw  = np.zeros((35,len(list_epoch_lengths),20,63),dtype=object)
    acz  = np.zeros((35,len(list_epoch_lengths),20,63),dtype=object)

    max_len=[]
    out_index=np.zeros((35,1),dtype=object)

    for subject in range(len(f_names)):
        filename = f_names[subject]
        skip=[]
        dat = read_mat(filename)['EEG_rest']

        ''' delete all non 9 and 32 makers'''
        myIndices = [i for i, x in enumerate(dat['event']['type']) if x in ['S 32','S  9']]
        dat['event']['type'] = [dat['event']['type'][i] for i in myIndices]
        dat['event']['latency'] = [dat['event']['latency'][i] for i in myIndices]

        if subject == 20:   #for some reason sub 20 has some more event markers at the end of the trial
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
        out_all=[]      #need this to read into extraction.py to kick out the thougt probes via the correct index
        out_long = []
        counter=0
        for i in range(len(dat['event']['type'])):
            if dat['event']['type'][i] == 9:
                single_interpretation = np.array((int(dat['event']['latency'][i]/500),0,int(dat['event']['type'][i]),i))
                single = np.array((int(dat['event']['latency'][i]),0,int(dat['event']['type'][i])))
                events_interpretation = np.append(events_interpretation,[single_interpretation],axis=0)
                events = np.append(events,[single],axis=0)
                counter+=1
                out_all.append(counter)
            if dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 10:
                out_long.append(counter)

        ch_names = dat['chanlocs']['labels']
        info = mne.create_info( ch_names=ch_names,
                                sfreq    = dat['srate'],
                                ch_types = ['eeg']*int(dat['nbchan']))
        raw = mne.io.RawArray(dat['data'],info)
        raw.filter(0.5, low_pass, verbose=False)

        event_ids = {'probe':9}
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
        baseline = (None, 0.0)

        acw_all_epoch_lens = np.zeros((len(list_epoch_lengths),20,63))
        for i,epoch_len in enumerate(list_epoch_lengths):
            
            tmin ,tmax = epoch_len,0         
            epochs =  mne.Epochs(raw, events=events, event_id=event_ids, tmin=tmin, tmax=tmax , baseline=baseline, picks=picks)  
        
            EEG = epochs.get_data()
            print(EEG.shape)
            skip=outs[i][subject][0]
            acw_return = sub_ACW(EEG,500,epoched=True,skip=skip)
            acw[subject,i] = acw_return[0]
            acz[subject,i] = acw_return[1]

    np.save('./data/data_generated/10_acw_05-{}'.format(str(low_pass)),acw)
    np.save('./data/data_generated/11_acz_05-{}'.format(str(low_pass)),acz)
