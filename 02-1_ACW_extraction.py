import numpy as np
import pandas as pd
from pymatreader import read_mat

freqs_low_pass =[40]

for low_pass in freqs_low_pass:
    ch_names = dat = read_mat('../data/data_raw/resting_state_thought_probe/Sub_35.mat')['EEG_rest']['chanlocs']['labels']
    acw = np.load('../data/data_generated/long_interval/acw_05-{}.npy'.format(str(low_pass)),allow_pickle=True)
    acz = np.load('../data/data_generated/long_interval/acz_05-{}.npy'.format(str(low_pass)),allow_pickle=True)
    outs = np.load('../data/data_generated/out_index.npy',allow_pickle=True)

    probe1 = read_mat('../data/data_raw/resting_state_thought_probe/resting_probe1.mat')
    probe1 = probe1['resting_probe1'].T

    all_col_names = ['subject','trial']
    n_intervals=[3,4,5,10,15,20,25,30]

    df_master = pd.DataFrame()
    for electrode in ch_names:
        for interval in n_intervals:
            temp_name = electrode+'_acw_'+str(interval)+'s'
            all_col_names.append(temp_name)
    df_master = pd.DataFrame()

    df  = pd.DataFrame(columns = all_col_names,index = np.arange(0,acw[0].shape[1]))    
    for subject in range(35):
        print('subject',subject)
        sub_acw = acw[subject]
        sub_acz = acz[subject]
        out=[]
        for i in range(20):
            if isinstance(sub_acw[0][i][0],np.int64) == True:
                out.append(i)
        for interval in range(len(n_intervals)):
            seconds=n_intervals[interval]
            for epoch in range(20):
                df.loc[epoch,'subject'] = int(subject+1)
                df.loc[epoch,'trial'] = int(epoch+1)
                for electrode in range(63):
                    col_name_acw = ch_names[electrode]+'_acw_'+str(seconds)+'s'
                    col_name_acz = ch_names[electrode]+'_acz_'+str(seconds)+'s'
                    if seconds < 10:
                        df.loc[epoch,col_name_acw] = acw[subject][interval][epoch][electrode]    
                        df.loc[epoch,col_name_acz] = acz[subject][interval][epoch][electrode]    
                    elif seconds > 5:              
                        if epoch not in outs[subject][0]:
                            df.loc[epoch,col_name_acw] = acw[subject][interval][epoch][electrode]
                            df.loc[epoch,col_name_acz] = acz[subject][interval][epoch][electrode]
                        else:
                            df.loc[epoch,col_name_acw] = np.nan 
                            df.loc[epoch,col_name_acz] = np.nan 
        df['probe1'] = probe1[subject,:].tolist()
        df_master = pd.concat([df_master,df])

    df_master.to_csv('./data/data_generated/12_ACW_all_subjects_05-{}.csv'.format(str(low_pass)))
