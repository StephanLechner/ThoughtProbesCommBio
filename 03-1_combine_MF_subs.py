import numpy as np
container = np.zeros((35,8,20,63))

for subject in range(35):
    sub = np.load('../data/data_generated/long_interval/sub_{}_mf_0.5_40.npy'.format(subject),allow_pickle=True)
    container[subject,:,:,:] = sub

np.save('./data/data_generated/21_mf_0.5_40_from_combined.npy',container)
