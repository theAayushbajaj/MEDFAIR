import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pydicom as dicom
import argparse

parser = argparse.ArgumentParser(description='COVID-CT data preprocessing')
parser.add_argument('--data_dir', type=str, help='path to the data folder')

args = parser.parse_args()

# read metadata
path = args.data_dir

demo_data = pd.read_csv(path + '/Clinical-data.csv', skiprows=1)
demo_data

Diagnosis_list = demo_data['Diagnosis'].values.tolist()
Folder_list = demo_data['Folder'].values.tolist()

Path_list = [x +'/'+y+'.npy' for x, y in zip(Diagnosis_list, Folder_list)]

binary_label_list = [1 if x=='COVID-19' else 0 for x in Diagnosis_list]

demo_data['Path'] = Path_list
demo_data['binary_label'] = binary_label_list


demo_data['Age_multi'] = demo_data['Patient Age'].str[:-1].values.astype('int')
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(-1,20), 0, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(20,39), 1, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(40,59), 2, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(60,79), 3, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi']>=80, 4, demo_data['Age_multi'])

demo_data['Age_binary'] = demo_data['Patient Age'].str[:-1].values.astype('int')
demo_data['Age_binary'] = np.where(demo_data['Age_binary'].between(-1, 60), 0, demo_data['Age_binary'])
demo_data['Age_binary'] = np.where(demo_data['Age_binary']>= 60, 1, demo_data['Age_binary'])

demo_data = demo_data.rename(columns={'Patient Gender': 'Sex'})

def split_712(all_meta, patient_ids):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.3, random_state=10)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.66, random_state=0)
    train_meta = all_meta[all_meta.Folder.isin(sub_train.astype('str'))]
    val_meta = all_meta[all_meta.Folder.isin(sub_val.astype('str'))]
    test_meta = all_meta[all_meta.Folder.isin(sub_test.astype('str'))]
    return train_meta, val_meta, test_meta

sub_train, sub_val, sub_test = split_712(demo_data, np.unique(demo_data['Folder']))

os.makedirs(path + '/split', exist_ok=True)

save_dir = path + '/split/'

sub_train.to_csv(save_dir + 'new_train.csv', index=False)
sub_val.to_csv(save_dir + 'new_val.csv', index=False)
sub_test.to_csv(save_dir + 'new_test.csv', index=False)


import pydicom
from PIL import Image
import pickle

train_meta = pd.read_csv(save_dir + 'new_train.csv')
val_meta = pd.read_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_val.csv')
test_meta = pd.read_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_test.csv')

def save_pkl(df, pkl_filename):
    images = []
    for i in range(len(df)):

        # Open the DICOM file
        dcm = pydicom.dcmread(df.iloc[i]['Path'])
        # Convert DICOM to numpy array
        img_array = dcm.pixel_array
        # Normalize to 0-255
        img_array = ((img_array - img_array.min()) * (1/(img_array.max() - img_array.min()) * 255)).astype('uint8')
        # Resize each slice to 256x256
        img_array = np.array([np.array(Image.fromarray(slice).resize((256, 256))) for slice in img_array])
        # Take the central 80 slices
        start = len(img_array) // 2 - 40
        img_array = img_array[start:start+80]
        # Random cropping of size 224x224x80 for training
        start_x = np.random.randint(0, img_array.shape[1] - 224)
        start_y = np.random.randint(0, img_array.shape[2] - 224)
        img_array = img_array[:, start_x:start_x+224, start_y:start_y+224]
        images.append(img_array)

    with open(path + f'{pkl_filename}.pkl', 'wb') as f:
        pickle.dump(images, f)

save_pkl(train_meta, 'train_images')
save_pkl(val_meta, 'val_images')
save_pkl(test_meta, 'test_images')