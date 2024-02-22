import glob
import os
from pymatreader import read_mat
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = '../data/data_raw/resting_state_thought_probe'
f_names=[]
for filename in sorted(glob.glob(os.path.join(path, 'Sub*'))):
    f_names.append(filename)
filename = f_names[0]
dat = read_mat(filename)['EEG_rest']

ch_names = read_mat(filename)['EEG_rest']['chanlocs']['labels']

df_corrR = pd.read_csv('./data/data_generated/df_corrR.csv',index_col='Unnamed: 0')
df_corrP = pd.read_csv('./data/data_generated/df_corrp.csv',index_col='Unnamed: 0')

columns_to_drop = df_corrR.columns[df_corrR.columns.str.contains('30')]
df_corrR = df_corrR.drop(columns=columns_to_drop, inplace=False)
columns_to_drop = df_corrP.columns[df_corrP.columns.str.contains('30')]
df_corrP = df_corrP.drop(columns=columns_to_drop, inplace=False)

df_corrR_mf   = df_corrR.filter(regex='mf')
df_corrR_acw  = df_corrR.filter(regex='acw')
df_corrR_acz  = df_corrR.filter(regex='acz')
df_corrP_mf   = df_corrP.filter(regex='mf')
df_corrP_acw  = df_corrP.filter(regex='acw')
df_corrP_acz  = df_corrP.filter(regex='acz')
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_corrR_mf  = df_corrR_mf.iloc[:,  ::-1]
df_corrR_acw = df_corrR_acw.iloc[:, ::-1]
df_corrR_acz = df_corrR_acz.iloc[:, ::-1]
df_corrP_mf  = df_corrP_mf.iloc[:,  ::-1]
df_corrP_acw = df_corrP_acw.iloc[:, ::-1]
df_corrP_acz = df_corrP_acz.iloc[:, ::-1]


df_corrP_mf[df_corrP_mf > 0.1]   = 0.11
df_corrP_acw[df_corrP_acw > 0.1] = 0.11
df_corrP_acz[df_corrP_acz > 0.1] = 0.11

subs = {
    'Fp1':'FP1',
    'Fpz':'FPZ', 
    'Fp2':'FP2',
    'Fz': 'FZ' ,
    'FCz':'FCZ', 
    'Cz': 'CZ' , 
    'CPz':'CPZ',
    'Pz': 'PZ' ,
    'POz':'POZ',
    'Oz' :'OZ' }

rev_subs = { v:k for k,v in subs.items()}
ch_names = [rev_subs.get(item,item)  for item in ch_names]

info = mne.create_info( ch_names=ch_names,
                        sfreq    = dat['srate'],
                        ch_types = ['eeg']*int(dat['nbchan']))
raw = mne.io.RawArray(dat['data'],info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

names_on = False

mf_max  = max([max(df_corrR_mf['mf_R_3']),max(df_corrR_mf['mf_R_4']),
               max(df_corrR_mf['mf_R_5']),max(df_corrR_mf['mf_R_10']),
               max(df_corrR_mf['mf_R_15']),max(df_corrR_mf['mf_R_20']),
               max(df_corrR_mf['mf_R_25'])])#,max(df_corrR_mf['mf_R_30'])])
               
mf_min  = min([min(df_corrR_mf['mf_R_3']),min(df_corrR_mf['mf_R_4']),
               min(df_corrR_mf['mf_R_5']),min(df_corrR_mf['mf_R_10']),
               min(df_corrR_mf['mf_R_15']),min(df_corrR_mf['mf_R_20']),
               min(df_corrR_mf['mf_R_25'])])#,min(df_corrR_mf['mf_R_30'])])

acw_max  = max([max(df_corrR_acw['acw_R_3']),max(df_corrR_acw['acw_R_4']),
               max(df_corrR_acw['acw_R_5']),max(df_corrR_acw['acw_R_10']),
               max(df_corrR_acw['acw_R_15']),max(df_corrR_acw['acw_R_20']),
               max(df_corrR_acw['acw_R_25'])])#,max(df_corrR_acw['acw_R_30'])])

acw_min  = min([min(df_corrR_acw['acw_R_3']),min(df_corrR_acw['acw_R_4']),
               min(df_corrR_acw['acw_R_5']),min(df_corrR_acw['acw_R_10']),
               min(df_corrR_acw['acw_R_15']),min(df_corrR_acw['acw_R_20']),
               min(df_corrR_acw['acw_R_25'])])#,min(df_corrR_acw['acw_R_30'])])

acz_max  = max([max(df_corrR_acz['acz_R_3']),max(df_corrR_acz['acz_R_4']),
               max(df_corrR_acz['acz_R_5']),max(df_corrR_acz['acz_R_10']),
               max(df_corrR_acz['acz_R_15']),max(df_corrR_acz['acz_R_20']),
               max(df_corrR_acz['acz_R_25'])])#,max(df_corrR_acz['acz_R_30'])])

acz_min  = min([min(df_corrR_acz['acz_R_3']),min(df_corrR_acz['acz_R_4']),
               min(df_corrR_acz['acz_R_5']),min(df_corrR_acz['acz_R_10']),
               min(df_corrR_acz['acz_R_15']),min(df_corrR_acz['acz_R_20']),
               min(df_corrR_acz['acz_R_25'])])#,min(df_corrR_acz['acz_R_30'])])

p_max = 0.05
p_min=0


fig,((mf25_R,mf20_R,mf15_R,mf10_R,mf5_R,mf4_R,mf3_R), 
(mf25_P,mf20_P,mf15_P,mf10_P,mf5_P,mf4_P,mf3_P,),
(acw25_R,acw20_R,acw15_R,acw10_R,acw5_R,acw4_R,acw3_R), 
(acw25_P,acw20_P,acw15_P,acw10_P,acw5_P,acw4_P,acw3_P),
(acz25_R,acz20_R,acz15_R,acz10_R,acz5_R,acz4_R,acz3_R), 
(acz25_P,acz20_P,acz15_P,acz10_P,acz5_P,acz4_P,acz3_P))   = plt.subplots(ncols=7,nrows=6,figsize=[18,18])

'''mark channels'''
mask_params = dict(markersize=4, markerfacecolor='#5C0DF0')

fz_mask = np.array(ch_names,dtype='bool')
cpz_mask = np.array(ch_names,dtype='bool')
x = ['F3','F4','Cz','Pz','O1','O2','Oz' ]
indices = [ch_names.index(element) if element in ch_names else None for element in x]
fz_mask[:]=False
fz_mask[indices]=True

green = sns.light_palette("seagreen", reverse=True, as_cmap=True)
green.set_over('gray')

mf1R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_3'], pos=raw.info,names=ch_names, axes=mf3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf2R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_4'], pos=raw.info,names=ch_names, axes=mf4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf3R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_5'], pos=raw.info,names=ch_names, axes=mf5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf4R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_10'],pos=raw.info,names=ch_names, axes=mf10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf5R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_15'],pos=raw.info,names=ch_names, axes=mf15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf6R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_20'],pos=raw.info,names=ch_names, axes=mf20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf7R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_25'],pos=raw.info,names=ch_names, axes=mf25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

mf1p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_3'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf3_P ,cmap=green,ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf2p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_4'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf4_P ,cmap=green,ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf3p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_5'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf5_P ,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf4p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_10'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf10_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf5p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_15'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf15_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf6p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_20'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf20_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf7p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_25'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf25_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acw1R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_3'], pos=raw.info,names=ch_names, axes=acw3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw2R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_4'], pos=raw.info,names=ch_names, axes=acw4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw3R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_5'], pos=raw.info,names=ch_names, axes=acw5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw4R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_10'],pos=raw.info,names=ch_names, axes=acw10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw5R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_15'],pos=raw.info,names=ch_names, axes=acw15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw6R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_20'],pos=raw.info,names=ch_names, axes=acw20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw7R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_25'],pos=raw.info,names=ch_names, axes=acw25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acw1p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_3'], pos=raw.info,names=ch_names, axes=acw3_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw2p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_4'], pos=raw.info,names=ch_names, axes=acw4_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw3p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_5'], pos=raw.info,names=ch_names, axes=acw5_P ,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw4p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_10'],pos=raw.info,names=ch_names, axes=acw10_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw5p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_15'],pos=raw.info,names=ch_names, axes=acw15_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw6p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_20'],pos=raw.info,names=ch_names, axes=acw20_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw7p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_25'],pos=raw.info,names=ch_names, axes=acw25_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acz1R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_3'], pos=raw.info,names=ch_names, axes=acz3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz2R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_4'], pos=raw.info,names=ch_names, axes=acz4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz3R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_5'], pos=raw.info,names=ch_names, axes=acz5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz4R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_10'],pos=raw.info,names=ch_names, axes=acz10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz5R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_15'],pos=raw.info,names=ch_names, axes=acz15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz6R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_20'],pos=raw.info,names=ch_names, axes=acz20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz7R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_25'],pos=raw.info,names=ch_names, axes=acz25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acz1p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_3'], pos=raw.info,names=ch_names, axes=acz3_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz2p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_4'], pos=raw.info,names=ch_names, axes=acz4_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz3p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_5'], pos=raw.info,names=ch_names, axes=acz5_P ,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz4p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_10'],pos=raw.info,names=ch_names, axes=acz10_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz5p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_15'],pos=raw.info,names=ch_names, axes=acz15_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz6p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_20'],pos=raw.info,names=ch_names, axes=acz20_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz7p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_25'],pos=raw.info,names=ch_names, axes=acz25_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

mf3_R.set_title(' MF\n 3s')
mf4_R.set_title(' MF\n 4s')
mf5_R.set_title(' MF\n 5s')
mf10_R.set_title('MF\n 10s')
mf15_R.set_title('MF\n 15s')
mf20_R.set_title('MF\n 20s')
mf25_R.set_title('MF\n 25s')

acw3_R.set_title(' ACW\n 3s')
acw4_R.set_title(' ACW\n 4s')
acw5_R.set_title(' ACW\n 5s')
acw10_R.set_title('ACW\n 10s')
acw15_R.set_title('ACW\n 15s')
acw20_R.set_title('ACW\n 20s')
acw25_R.set_title('ACW\n 25s')

acz3_R.set_title(' ACZ\n 3s')
acz4_R.set_title(' ACZ\n 4s')
acz5_R.set_title(' ACZ\n 5s')
acz10_R.set_title('ACZ\n 10s')
acz15_R.set_title('ACZ\n 15s')
acz20_R.set_title('ACZ\n 20s')
acz25_R.set_title('ACZ\n 25s')
plt.savefig('./plots/composite_figure/topomaps_turned1',dpi=800)

f_mf,((mf25_R,mf20_R,mf15_R,mf10_R,mf5_R,mf4_R,mf3_R), 
(mf25_P,mf20_P,mf15_P,mf10_P,mf5_P,mf4_P,mf3_P))   = plt.subplots(ncols=7,nrows=2,figsize=[11,3])

mf1R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_3'], pos=raw.info,names=ch_names, axes=mf3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf2R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_4'], pos=raw.info,names=ch_names, axes=mf4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf3R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_5'], pos=raw.info,names=ch_names, axes=mf5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf4R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_10'],pos=raw.info,names=ch_names, axes=mf10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf5R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_15'],pos=raw.info,names=ch_names, axes=mf15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf6R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_20'],pos=raw.info,names=ch_names, axes=mf20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf7R,cm = mne.viz.plot_topomap(df_corrR_mf['mf_R_25'],pos=raw.info,names=ch_names, axes=mf25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

mf1p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_3'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf3_P ,cmap=green,ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf2p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_4'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf4_P ,cmap=green,ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf3p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_5'], vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf5_P ,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf4p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_10'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf10_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf5p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_15'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf15_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf6p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_20'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf20_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
mf7p,cm = mne.viz.plot_topomap(df_corrP_mf['mf_p_25'],vlim=(0,0.01),pos=raw.info,names=ch_names, axes=mf25_P,cmap=green,ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

mf3_R.set_title(' MF\n 3s')
mf4_R.set_title(' MF\n 4s')
mf5_R.set_title(' MF\n 5s')
mf10_R.set_title('MF\n 10s')
mf15_R.set_title('MF\n 15s')
mf20_R.set_title('MF\n 20s')
mf25_R.set_title('MF\n 25s')
plt.tight_layout()
plt.savefig('./plots/composite_figure/topomaps_MF',dpi=800)

f_acw,((acw25_R,acw20_R,acw15_R,acw10_R,acw5_R,acw4_R,acw3_R), 
(acw25_P,acw20_P,acw15_P,acw10_P,acw5_P,acw4_P,acw3_P))   = plt.subplots(ncols=7,nrows=2,figsize=[11,3])

acw1R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_3'], pos=raw.info,names=ch_names, axes=acw3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw2R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_4'], pos=raw.info,names=ch_names, axes=acw4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw3R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_5'], pos=raw.info,names=ch_names, axes=acw5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw4R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_10'],pos=raw.info,names=ch_names, axes=acw10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw5R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_15'],pos=raw.info,names=ch_names, axes=acw15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw6R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_20'],pos=raw.info,names=ch_names, axes=acw20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw7R,cm = mne.viz.plot_topomap(df_corrR_acw['acw_R_25'],pos=raw.info,names=ch_names, axes=acw25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acw1p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_3'], pos=raw.info,names=ch_names, axes=acw3_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw2p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_4'], pos=raw.info,names=ch_names, axes=acw4_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw3p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_5'], pos=raw.info,names=ch_names, axes=acw5_P ,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw4p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_10'],pos=raw.info,names=ch_names, axes=acw10_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw5p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_15'],pos=raw.info,names=ch_names, axes=acw15_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw6p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_20'],pos=raw.info,names=ch_names, axes=acw20_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acw7p,cm = mne.viz.plot_topomap(df_corrP_acw['acw_p_25'],pos=raw.info,names=ch_names, axes=acw25_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acw3_R.set_title(' ACW\n 3s')
acw4_R.set_title(' ACW\n 4s')
acw5_R.set_title(' ACW\n 5s')
acw10_R.set_title('ACW\n 10s')
acw15_R.set_title('ACW\n 15s')
acw20_R.set_title('ACW\n 20s')
acw25_R.set_title('ACW\n 25s')
plt.tight_layout()
plt.savefig('./plots/composite_figure/topomaps_ACW',dpi=800)

f_acz,((acz25_R,acz20_R,acz15_R,acz10_R,acz5_R,acz4_R,acz3_R), 
(acz25_P,acz20_P,acz15_P,acz10_P,acz5_P,acz4_P,acz3_P))   = plt.subplots(ncols=7,nrows=2,figsize=[11,3])

acz1R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_3'], pos=raw.info,names=ch_names, axes=acz3_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz2R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_4'], pos=raw.info,names=ch_names, axes=acz4_R ,cmap='Spectral_r', ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz3R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_5'], pos=raw.info,names=ch_names, axes=acz5_R ,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz4R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_10'],pos=raw.info,names=ch_names, axes=acz10_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz5R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_15'],pos=raw.info,names=ch_names, axes=acz15_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz6R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_20'],pos=raw.info,names=ch_names, axes=acz20_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz7R,cm = mne.viz.plot_topomap(df_corrR_acz['acz_R_25'],pos=raw.info,names=ch_names, axes=acz25_R,cmap='Spectral_r', ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acz1p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_3'], pos=raw.info,names=ch_names, axes=acz3_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz2p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_4'], pos=raw.info,names=ch_names, axes=acz4_P ,cmap=green, ch_type='eeg', contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz3p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_5'], pos=raw.info,names=ch_names, axes=acz5_P ,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz4p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_10'],pos=raw.info,names=ch_names, axes=acz10_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz5p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_15'],pos=raw.info,names=ch_names, axes=acz15_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz6p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_20'],pos=raw.info,names=ch_names, axes=acz20_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)
acz7p,cm = mne.viz.plot_topomap(df_corrP_acz['acz_p_25'],pos=raw.info,names=ch_names, axes=acz25_P,cmap=green, ch_type='eeg',  contours=4, show=False,mask=fz_mask, mask_params=mask_params)

acz3_R.set_title(' ACZ\n 3s')
acz4_R.set_title(' ACZ\n 4s')
acz5_R.set_title(' ACZ\n 5s')
acz10_R.set_title('ACZ\n 10s')
acz15_R.set_title('ACZ\n 15s')
acz20_R.set_title('ACZ\n 20s')
acz25_R.set_title('ACZ\n 25s')
plt.tight_layout()
plt.savefig('./plots/composite_figure/topomaps_ACZ',dpi=800)

''' H E A T M A P S'''
plt.figure(figsize=(18,18))
selected_electrodes = ['F3','F4','Pz','O1','O2','Oz']

df_corrR_mf  = df_corrR_mf.filter(items  = selected_electrodes, axis=0)
df_corrR_acw = df_corrR_acw.filter(items = selected_electrodes, axis=0)
df_corrR_acz = df_corrR_acz.filter(items = selected_electrodes, axis=0)
df_corrP_mf  = df_corrP_mf.filter(items  = selected_electrodes, axis=0)
df_corrP_acw = df_corrP_acw.filter(items = selected_electrodes, axis=0)
df_corrP_acz = df_corrP_acz.filter(items = selected_electrodes, axis=0)

green = sns.light_palette("seagreen", reverse=True, as_cmap=True)
green.set_over('gray')

fig, (ax1,ax2) = plt.subplots(1,2,figsize=[9,3]) #  25,15

sns.heatmap(df_corrR_mf, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5.5}, fmt='.2f',cmap='bwr',vmin=-0.4, vmax=0.4,ax=ax1)
ax1.tick_params(axis='y', rotation=0.1)
g2 = sns.heatmap(df_corrP_mf, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5}, fmt='.3f',cmap=green, vmin=0, vmax=0.05, cbar_kws={'extend': 'max'},ax=ax2)
g2.set(yticklabels=[])
g2.set(xlabel=None)
ax1.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
ax2.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
plt.tight_layout()
plt.savefig('./plots/composite_figure/1_mf_heatmap_turned1',dpi=800)
plt.clf()
plt.close()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=[9,3]) #  25,15
sns.heatmap(df_corrR_acw, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5.5}, fmt='.2f',cmap='bwr',vmin=-0.4, vmax=0.4,ax=ax1)
ax1.tick_params(axis='y', rotation=0.1)
g2 = sns.heatmap(df_corrP_acw, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5}, fmt='.3f',cmap=green, vmin=0, vmax=0.05, cbar_kws={'extend': 'max'},ax=ax2)
g2.set(yticklabels=[])
g2.set(xlabel=None)
ax1.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
ax2.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
plt.tight_layout()
plt.savefig('./plots/composite_figure/2_acw_heatmap_turned1',dpi=800)
plt.clf()
plt.close()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=[9,3]) #  25,15
sns.heatmap(df_corrR_acz, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5.5}, fmt='.2f',cmap='bwr',vmin=-0.4, vmax=0.4,ax=ax1)
ax1.tick_params(axis='y', rotation=0.1)
g2 = sns.heatmap(df_corrP_acz, square=True, linewidths=.5, annot=True, annot_kws={"fontsize":5}, fmt='.3f',cmap=green, vmin=0, vmax=0.05, cbar_kws={'extend': 'max'},ax=ax2)
g2.set(yticklabels=[])
g2.set(xlabel=None)
ax1.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
ax2.set_xticklabels(['25s','20s','15s','10s','5s','4s','3s'])
plt.tight_layout()
plt.savefig('./plots/composite_figure/3_acz_heatmap_turned1',dpi=800)
plt.clf()
plt.close()


''' LINE PLOTS '''
plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrR_mf.loc[chan]
    plt.plot(x,label=chan)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_1_df_corrR_mf_turned1',dpi=800)
plt.clf()
plt.close()

plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrP_mf.loc[chan]
    plt.plot(x,label=chan)
plt.axhline(0.05,linestyle='--',color='red')
plt.axhline(0.1,linestyle='-',color='gray',alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_2_df_corrp_mf_turned1',dpi=800)
plt.clf()
plt.close()

plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrR_acw.loc[chan]
    plt.plot(x,label=chan)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_3_df_corrR_acw_turned1',dpi=800)
plt.clf()
plt.close()

plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrP_acw.loc[chan]
    plt.plot(x,label=chan)
plt.axhline(0.05,linestyle='--',color='red')
plt.axhline(0.1,linestyle='-',color='gray',alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_4_df_corrp_acw_turned1',dpi=800)
plt.clf()
plt.close()

plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrR_acz.loc[chan]
    plt.plot(x,label=chan)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_5_df_corrR_acz_turned1',dpi=800)
plt.clf()
plt.close()

plt.figure(figsize=(18,2.5))
for chan in selected_electrodes:
    x = df_corrP_acz.loc[chan]
    plt.plot(x,label=chan)
plt.axhline(0.05,linestyle='--',color='red')
plt.axhline(0.1,linestyle='-',color='gray',alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6,],labels=['25s','20s','15s','10s','5s','4s','3s'])
plt.savefig('./plots/composite_figure/lineplot_6_df_corrp_acz_turned1',dpi=800)
plt.clf()
plt.close()
