#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch

# Load Raw Data Dict
raw_data = pickle.load(open("RML2016.10a_dict.pkl", 'rb'), encoding='latin1')

modulations = []
for key in raw_data.keys():
    modulations.append(key[0])

unique_modulations = list(set(modulations))

# Create an "index" which is just a list of tuples describe each piece of data
# Also, create a raw-binary file that we can index into to get each piece of IQ data.
# This facilitates usage of a Python generator that is more friendly to systems with
# lower memory.
with open("rml_data.f32", 'ab+') as data_file:
    index = []
    for key_idx, key in enumerate(raw_data.keys()):
        for data_idx in range(len(raw_data[key])):
            index_number = key_idx*1000 + data_idx
            modulation = key[0]
            snr = key[1]
            iq_data = raw_data[key][data_idx]
            item_size = iq_data.dtype.itemsize
            num_elements = iq_data.size
            data_offset = index_number*num_elements * \
                item_size  # number of bytes into the data
            index.append((index_number, modulation, unique_modulations.index(
                modulation), snr, data_offset))
            iq_data.flatten().tofile(data_file)

with open("rml_index.pkl", 'wb') as index_file:
    pickle.dump(index, index_file)


# Just as example, load the data from the file we just made and index into it.
dataset_index = pickle.load(open("rml_index.pkl", 'rb'))
item_index = 4000
data = np.fromfile("rml_data.f32", dtype=np.float32, count=256,
                   offset=dataset_index[item_index][4]).reshape(2, 128)
data2 = np.fromfile("rml_data.f32", dtype=np.float32, count=256,
                    offset=dataset_index[item_index + 1][4]).reshape(2, 128)
mini_batch = torch.stack((torch.from_numpy(data), torch.from_numpy(data2)), 0)
label = dataset_index[item_index][2]

# Plot it.
plt.plot(mini_batch[0, 0])
plt.plot(mini_batch[0, 1])
print(dataset_index[item_index])
