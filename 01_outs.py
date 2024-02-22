import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) 
import pandas as pd
from pymatreader import read_mat
import glob
import os
from sys import exit
import seaborn as sns

f_names = []
ndar = []
path = '../../data/data_raw/resting_state_thought_probe/'
for filename in sorted(glob.glob(os.path.join(path, 'Sub*'))):
    f_names.append(filename)
f_names
max_len=[]

trial_durations = np.zeros((35,20))

out_index=np.zeros((8,35,1),dtype=object)
for subject in range(len(f_names)):

    filename = f_names[subject]
    dat = read_mat(filename)['EEG_rest']

    ''' delete all non 9 and 32 makers'''
    myIndices = [i for i, x in enumerate(dat['event']['type']) if x in ['S 32','S  9']]
    dat['event']['type'] = [dat['event']['type'][i] for i in myIndices]
    dat['event']['latency'] = [dat['event']['latency'][i] for i in myIndices]

    if subject == 20:   #for some reason sub 20 has some more event markers at the end of the trial. just take the first 20
        dat['event']['type'] = dat['event']['type'][:40]
        dat['event']['latency'] = dat['event']['latency'][:40]

    for i in range(len(dat['event']['type'])):
        if dat['event']['type'][i] == 'S 32':
            dat['event']['type'][i]= 32
        elif dat['event']['type'][i] == 'S  9':
            dat['event']['type'][i]= 9
    markers=dat['event']['type']


    print(markers.count(9),markers.count(32))
    fig,ax = plt.subplots(1,1)
    lat = dat['event']['latency']
    lat = [x / 500 for x in dat['event']['latency']]
    max_len.append(lat[-1])
    counter=0
    for i in range(len(lat)):
        if markers[i] == 32:
            ax.axvline(lat[i],color='green',linestyle='--')
            my32=lat[i]
            next32=lat[i+1]

        elif markers[i] == 9:
            ax.axvline(lat[i],color='orange')
            my9 = lat[i]
            print('trial',counter,'\t',round(my9-my32) ,'\t= time between trial start and probe')
            trial_durations[subject,counter] = round(my9-my32)
            counter+=1 
            if counter>20:
                break    

    events_interpretation       = np.empty((0,4), int)
    events                      = np.empty((0,3), int)
    out_all=[]      #need this to read into extraction.py to kick out the thougt probes via the correct index
    counter=0
    out_smaller5 = []
    out_smaller10 = []
    out_smaller15 = []
    out_smaller20 = []
    out_smaller25 = []
    out_smaller30 = []
    for i in range(len(dat['event']['type'])):
        if dat['event']['type'][i] == 9:
            single_interpretation = np.array((int(dat['event']['latency'][i]/500),0,int(dat['event']['type'][i]),i))
            single = np.array((int(dat['event']['latency'][i]),0,int(dat['event']['type'][i])))
            events_interpretation = np.append(events_interpretation,[single_interpretation],axis=0)
            events = np.append(events,[single],axis=0)
            counter+=1
            out_all.append(counter-1)
            print(counter-1,'single_interpretation',single_interpretation,'\tsingle',single,'\t',dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500)
        # if  dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) < 30 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 25:
        if  dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) <30: #< 30 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 25:
            out_smaller30.append(counter-1)
            print('\t',np.round(dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500,1),'is smaller than 30 seconds')            # counter+=1
        if dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) < 25:# and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 20:
            out_smaller25.append(counter-1)
            print('\t',np.round(dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500,1),'is smaller than 25 seconds')            # counter+=1
        if dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) < 20:# and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 15:
            out_smaller20.append(counter-1)
            print('\t',np.round(dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500,1),'is smaller than 20 seconds')            # counter+=1
        if dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) < 15:# and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 10:
            out_smaller15.append(counter-1)
            print('\t',np.round(dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500,1),'is smaller than 15 seconds')            # counter+=1
        if dat['event']['type'][i] == 9 and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) < 10:# and (dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500) > 5:
            out_smaller10.append(counter-1   )
            print('\t',np.round(dat['event']['latency'][i]/500-dat['event']['latency'][i-1]/500,1),'is smaller than 10 seconds')
    print('<10',out_smaller10)
    print('<15',out_smaller15)
    print('<20',out_smaller20)
    print('<25',out_smaller25)
    print('<30',out_smaller30)
    # input()
            # input('smaller10')
    out_list30 = list(set(out_all) - set(out_smaller30))
    out_list25 = list(set(out_all) - set(out_smaller25))
    out_list20 = list(set(out_all) - set(out_smaller20))
    out_list15 = list(set(out_all) - set(out_smaller15))
    out_list10 = list(set(out_all) - set(out_smaller10))

    out_index[0][subject][0]=[]
    out_index[1][subject][0]=[]
    out_index[2][subject][0]=[]
    out_index[3][subject][0]=out_smaller10
    out_index[4][subject][0]=out_smaller15
    out_index[5][subject][0]=out_smaller20
    out_index[6][subject][0]=out_smaller25
    out_index[7][subject][0]=out_smaller30


df = pd.DataFrame(trial_durations.T)
plt.figure(figsize=(18, 6))  # Adjust the figure size if needed
sns.stripplot(data=df, jitter=0.4, size=5, alpha=0.7,)
for x in range(len(df.columns)):
    plt.axvline(x - 0.5, color='gray', linestyle='--', lw=0.5)
plt.xlabel('Subject')
plt.ylabel('Trial Durations (s)')
plt.title('Trial durations')

custom_yticks = [5, 15, 25, 35, 45, 55, 65]
plt.yticks(custom_yticks)

plt.ylim(0,70)
plt.margins(0.01)  # Adjust the value as needed
plt.tight_layout(pad=0.1)  # Adjust the pad value as needed
plt.savefig('./plots/trial durations',dpi=500)
