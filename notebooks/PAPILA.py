import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse
import pickle
from PIL import Image

parser = argparse.ArgumentParser(description='PAPILA data preprocessing')
parser.add_argument('--data_dir', type=str, help='path to the data folder')
args = parser.parse_args()
# read metadata
path = args.data_dir

# OD for right, OS for left
od_meta = pd.read_csv(path + 'ClinicalData/patient_data_od.csv')
os_meta = pd.read_csv(path + 'ClinicalData/patient_data_os.csv')

ids = os_meta['ID'].values
os_path = ['RET' + x[1:] + 'OS.jpg' for x in ids]
os_meta['Path'] = os_path

ids = od_meta['ID'].values
od_path = ['RET' + x[1:] + 'OD.jpg' for x in ids]
od_meta['Path'] = od_path

meta_all = pd.concat([od_meta, os_meta])
subcolumns = ['ID', 'Age', 'Gender', 'Diagnosis', 'Path']
meta_all = meta_all[subcolumns]

meta_all.to_csv(path + 'ClinicalData/patient_meta_concat.csv')

# the patient (0 for male and 1 for female), 
# the diagnosis (0 stands for healthy, 1 for glaucoma, and 2 for suspicious)

sex = meta_all['Gender'].values.astype('str')
sex[sex == '0.0'] = 'M'
sex[sex == '1.0'] = 'F'
meta_all['Sex'] = sex

meta_all['Age_multi'] = meta_all['Age'].values.astype('int')
meta_all['Age_multi'] = np.where(meta_all['Age_multi'].between(0,19), 0, meta_all['Age_multi'])
meta_all['Age_multi'] = np.where(meta_all['Age_multi'].between(20,39), 1, meta_all['Age_multi'])
meta_all['Age_multi'] = np.where(meta_all['Age_multi'].between(40,59), 2, meta_all['Age_multi'])
meta_all['Age_multi'] = np.where(meta_all['Age_multi'].between(60,79), 3, meta_all['Age_multi'])
meta_all['Age_multi'] = np.where(meta_all['Age_multi']>=80, 4, meta_all['Age_multi'])

meta_all['Age_binary'] = meta_all['Age'].values.astype('int')
meta_all['Age_binary'] = np.where(meta_all['Age_binary'].between(0, 60), 0, meta_all['Age_binary'])
meta_all['Age_binary'] = np.where(meta_all['Age_binary']>= 60, 1, meta_all['Age_binary'])

# binary , only use healthy and glaucoma, i.e. 0 and 1.

meta_binary = meta_all[(meta_all['Diagnosis'].values == 1.0) | (meta_all['Diagnosis'].values == 0.0)]

def split_712(all_meta, patient_ids):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.3, random_state=5)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.66, random_state=15)
    train_meta = all_meta[all_meta.ID.isin(sub_train)]
    val_meta = all_meta[all_meta.ID.isin(sub_val)]
    test_meta = all_meta[all_meta.ID.isin(sub_test)]
    return train_meta, val_meta, test_meta

sub_train, sub_val, sub_test = split_712(meta_binary, np.unique(meta_binary['ID']))

os.makedirs(path + '/split', exist_ok=True)

save_dir = path + '/split/'

sub_train.to_csv(save_dir + 'new_train.csv', index=False)
sub_val.to_csv(save_dir + 'new_val.csv', index=False)
sub_test.to_csv(save_dir + 'new_test.csv', index=False)

train_meta = pd.read_csv(save_dir + 'new_train.csv')
val_meta = pd.read_csv(save_dir + 'new_val.csv')
test_meta = pd.read_csv(save_dir + 'new_test.csv')

os.makedirs(path + '/split', exist_ok=True)
pkl_path = path + '/pkls/'

os.makedirs(pkl_path, exist_ok=True)

def save_pkl(df, pkl_filename):
    images = []
    for i in range(len(df)):
        # Open the image file
        img = Image.open(path + '/FundusImages/' + df.iloc[i]['Path'])
        # Resize the image
        img = img.resize((256, 256))
        # Convert the image to a numpy array if necessary
        img_array = np.array(img)
        images.append(img_array)

    with open(path + f'{pkl_filename}.pkl', 'wb') as f:
        pickle.dump(images, f)

save_pkl(train_meta, 'train_images')
save_pkl(val_meta, 'val_images')
save_pkl(test_meta, 'test_images')
print('done')