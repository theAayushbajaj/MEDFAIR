# %%
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time


# %% [markdown]
# ## Preprocess metadata

# %%
# read metadata
path = '/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/'

demo_data = pd.read_csv(path + 'HAM10000_metadata.csv')

# %%
#Counter(demo_data['dataset'])

# %%
# add image path to the metadata
pathlist = demo_data['image_id'].values.tolist()
paths = ['/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/HAM10000_images/' + i + '.jpg' for i in pathlist]
demo_data['Path'] = paths

# %%
# remove age/sex == null 
demo_data = demo_data[~demo_data['age'].isnull()]
demo_data = demo_data[~demo_data['sex'].isnull()]

# %%
# unify the value of sensitive attributes
sex = demo_data['sex'].values
sex[sex == 'male'] = 'M'
sex[sex == 'female'] = 'F'
demo_data['Sex'] = sex

# %%
# split subjects to different age groups
demo_data['Age_multi'] = demo_data['age'].values.astype('int')
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(-1,19), 0, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(20,39), 1, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(40,59), 2, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(60,79), 3, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi']>=80, 4, demo_data['Age_multi'])

demo_data['Age_binary'] = demo_data['age'].values.astype('int')
demo_data['Age_binary'] = np.where(demo_data['Age_binary'].between(-1, 60), 0, demo_data['Age_binary'])
demo_data['Age_binary'] = np.where(demo_data['Age_binary']>= 60, 1, demo_data['Age_binary'])

# %%
# convert to binary labels
# benign: bcc, bkl, dermatofibroma, nv, vasc
# maglinant: akiec, mel

labels = demo_data['dx'].values.copy()
labels[labels == 'akiec'] = '1'
labels[labels == 'mel'] = '1'
labels[labels != '1'] = '0'

labels = labels.astype('int')

demo_data['binaryLabel'] = labels

# %% [markdown]
# ## Split train/val/test

# %%
def split_811(all_meta, patient_ids):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.2, random_state=0)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.5, random_state=0)
    train_meta = all_meta[all_meta.lesion_id.isin(sub_train)]
    val_meta = all_meta[all_meta.lesion_id.isin(sub_val)]
    test_meta = all_meta[all_meta.lesion_id.isin(sub_test)]
    return train_meta, val_meta, test_meta

sub_train, sub_val, sub_test = split_811(demo_data, np.unique(demo_data['lesion_id']))
print('ckpt 1')

# %%
os.makedirs('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split',exist_ok=True)
sub_train.to_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_train.csv')
sub_val.to_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_val.csv')
sub_test.to_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_test.csv')

print('ckpt 2')
# %%
# you can have a look of some examples here
#img = cv2.imread('your_path/fariness_data/HAM10000/HAM10000_images/ISIC_0027419.jpg')
#print(img.shape)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %% [markdown]
# ## Save images into pickle files
# This is optional, but if you are training many models, this step can save a lot of time by reducing the data IO.

# %%
train_meta = pd.read_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_train.csv')
val_meta = pd.read_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_val.csv')
test_meta = pd.read_csv('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/split/new_test.csv')

os.makedirs('/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/pkls', exist_ok=True)
path = '/home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/pkls/'

def save_pkl(df, pkl_filename):
    images = []
    start = time.time()
    for i in range(len(df)):

        # Open the image file
        img = Image.open(df.iloc[i]['Path'])
        # Resize the image
        img = img.resize((256, 256))
        # Convert the image to a numpy array if necessary
        img_array = np.array(img)
        images.append(img_array)

    end = time.time()
    end-start
    with open(path + f'{pkl_filename}.pkl', 'wb') as f:
        pickle.dump(images, f)

save_pkl(train_meta, 'train_images')
save_pkl(val_meta, 'val_images')
save_pkl(test_meta, 'test_images')
print('done')
