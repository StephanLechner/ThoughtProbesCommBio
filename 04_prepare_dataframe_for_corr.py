import pandas as pd
import numpy as np
from numpy import mean, std 
from math import sqrt
from pymatreader import read_mat
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=None)

def cohen_d(x,y):
        return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)  

ch_names = dat = read_mat('../data/data_raw/resting_state_thought_probe/Sub_35.mat')['EEG_rest']['chanlocs']['labels']
mf = np.load('./data/data_generated/21_mf_0.5_40_from_combined.npy',allow_pickle=True)

acw = np.load('./data/data_generated/10_acw_05-40.npy',allow_pickle=True)
acz = np.load('./data/data_generated/11_acz_05-40.npy',allow_pickle=True)
outs = np.load('./data/data_generated/00_out_index5-30.npy',allow_pickle=True)

probe1 = read_mat('../data/data_raw/resting_state_thought_probe/resting_probe1.mat')
probe1 = probe1['resting_probe1'].T

'''   QAUTNIFICATION OF PROBES'''
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator
df = pd.DataFrame(probe1.T)

df = df.apply(lambda x: x.sort_values(ascending=False).reset_index(drop=True))
df['probe']=['deliberate']*10+['spontaneous']*10

upper = df.iloc[:10,:-1].values.flatten().tolist()
lower = df.iloc[10:,:-1].values.flatten().tolist()

values = upper+lower
group = ['deliberate']*350+['spontaneous']*350
df_new = pd.DataFrame(values,columns=['values'])
df_new['probe'] = group

cm = 1/2.54  # centimeters in inches
f,ax = plt.subplots(1,1,figsize=(2+1*cm,4.7*1*cm),dpi=800)
flierprops = dict(markersize=2,  markeredgecolor='black')
ax_hue_plot_params = {'data': df_new,'x': 'probe','y': 'values',"palette": 'viridis',"width":0.5,"linewidth":0.9,"flierprops":flierprops}
ax = sns.boxplot(ax=ax,**ax_hue_plot_params)
plt.xlabel('')
plt.ylabel('Probe ratings',fontsize=10)

pairs = [("deliberate","spontaneous")]
ax_formatted_pvalues = ['****']

annotator = Annotator(ax, pairs, **ax_hue_plot_params,verbose=False,fontsize=0.2)
annotator.configure(loc='inside',line_width=0.7,fontsize=7).set_custom_annotations(ax_formatted_pvalues).annotate()
_, xlabels = plt.xticks()
ax.set_xticklabels(xlabels, size=8.5)
plt.tight_layout()
plt.savefig('./plots/median split probes behavioral1',dpi=400)


''' INTER SUBJECT CV'''
df = pd.DataFrame(probe1.T)
mean_values = df.mean(axis=0)  
std_dev_values = df.std(axis=0) 
cv_values = (std_dev_values / mean_values) * 100  # as a percentage
cv_df = pd.DataFrame({'Participant': df.columns, 'CV (%)': cv_values})
def classify_cv(cv):
    if cv < 10:
        return 'Low'
    elif cv < 20:
        return 'Moderate'
    else:
        return 'High'

cv_df['Classification'] = cv_df['CV (%)'].apply(classify_cv)
''''''

probe1 = probe1.reshape(700,1).flatten()
probe1 = probe1.flatten()

all_col_names = ['subject','trial']
n_intervals=[3,4,5,10,15,20,25,30]
n_interval_names = ['3s','4s','5s','10s','15s','20s','25s','30s']
corrRp = pd.DataFrame()
corrR  = pd.DataFrame()
corrp  = pd.DataFrame()

master_df=pd.DataFrame()

for interval in range(len(n_intervals)):
    count=0
    store = np.zeros((3,35*20,63))
    seconds=n_intervals[interval]
    for subject in range(35):
        for epoch in range(20):
            for electrode in range(63):

                if seconds > 5:      
                    if epoch in outs[interval][subject][0]:
                        mf[subject,interval,epoch,electrode] = np.nan
                        acw[subject,interval,epoch,electrode] = np.nan
                        acz[subject,interval,epoch,electrode] = np.nan
                store[0,count,electrode] = mf[subject,interval,epoch,electrode]
                store[1,count,electrode] = acw[subject,interval,epoch,electrode]        
                store[2,count,electrode] = acz[subject,interval,epoch,electrode]        
            count+=1

    df_mf = pd.DataFrame()
    df_mf['probe'] = probe1
    df_acw = pd.DataFrame()
    df_acw['probe'] = probe1
    df_acz = pd.DataFrame()
    df_acz['probe'] = probe1

    for electrode in range(63):
        ch_name = ch_names[electrode]
        mf_temp  = store[0,:,electrode]

        acw_temp = store[1,:,electrode]
        acz_temp = store[2,:,electrode]
        df_mf[ch_name] = mf_temp.tolist()

        df_acw[ch_name] = acw_temp.tolist()
        df_acz[ch_name] = acz_temp.tolist()

        master_df[ch_name+'_mf_'+n_interval_names[interval]]  = mf_temp.tolist()
        master_df[ch_name+'_acw_'+n_interval_names[interval]] = acw_temp.tolist()
        master_df[ch_name+'_acz_'+n_interval_names[interval]] = acz_temp.tolist()

        temp_df_mf = df_mf.dropna().reset_index(drop=True)
        temp_df_acw = df_acw.dropna().reset_index(drop=True)
        temp_df_acz = df_acz.dropna().reset_index(drop=True)

        col_name_R= 'R_'+str(n_intervals[interval])
        col_name_p= 'p_'+str(n_intervals[interval])
        corrRp.at[ch_name,'mf_'+col_name_R] = stats.pearsonr(temp_df_mf[ch_name],temp_df_mf['probe'])[0]
        corrRp.at[ch_name,'mf_'+col_name_p] = stats.pearsonr(temp_df_mf[ch_name],temp_df_mf['probe'])[1]    
        corrR.at[ch_name,'mf_'+col_name_R] = stats.pearsonr(temp_df_mf[ch_name],temp_df_mf['probe'])[0]
        corrp.at[ch_name,'mf_'+col_name_p] = stats.pearsonr(temp_df_mf[ch_name],temp_df_mf['probe'])[1]    

        corrRp.at[ch_name,'acw_'+col_name_R] = stats.pearsonr(temp_df_acw[ch_name],temp_df_acw['probe'])[0]
        corrRp.at[ch_name,'acw_'+col_name_p] = stats.pearsonr(temp_df_acw[ch_name],temp_df_acw['probe'])[1]    
        corrR.at[ch_name,'acw_'+col_name_R] = stats.pearsonr(temp_df_acw[ch_name],temp_df_acw['probe'])[0]
        corrp.at[ch_name,'acw_'+col_name_p] = stats.pearsonr(temp_df_acw[ch_name],temp_df_acw['probe'])[1]    

        corrRp.at[ch_name,'acz_'+col_name_R] = stats.pearsonr(temp_df_acz[ch_name],temp_df_acz['probe'])[0]
        corrRp.at[ch_name,'acz_'+col_name_p] = stats.pearsonr(temp_df_acz[ch_name],temp_df_acz['probe'])[1]    
        corrR.at[ch_name,'acz_'+col_name_R] = stats.pearsonr(temp_df_acz[ch_name],temp_df_acz['probe'])[0]
        corrp.at[ch_name,'acz_'+col_name_p] = stats.pearsonr(temp_df_acz[ch_name],temp_df_acz['probe'])[1]    

import statsmodels
def adjust_p(df):
    for i in range(len(df.columns)):
        col = df[df.columns[i]]
        col_name = df.columns[i]
        if 'p_' in col.name:
            pvals=col.tolist()
            p_vals_corr = statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
            df[col_name] = p_vals_corr
    return df

coorRp = adjust_p(corrRp)
coorp = adjust_p(corrp)
master_df.to_csv('./data/data_generated/df_for_mediation1.csv')
corrRp.to_csv('./data/data_generated/df_corrRp.csv',index=True)
corrR.to_csv('./data/data_generated/df_corrR.csv',index=True)
corrp.to_csv('./data/data_generated/df_corrp.csv',index=True)








