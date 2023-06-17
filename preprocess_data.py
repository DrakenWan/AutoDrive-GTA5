import numpy as np
from tqdm import tqdm

## LOADING DATA
# path and file names
dir_path = './Data/'
img_file_name = 'train_images.npy'
labels_file_name = 'train_labels.npy'

print("Loading our dataset...")
# looading our model data
images = np.load(dir_path + img_file_name, allow_pickle=True)
labels = np.load(dir_path + labels_file_name, allow_pickle=True)


print("Preprocess start...")
stack_imgs = np.empty((len(images),) + images[0].shape, dtype='int8')

print("Stacking image data.")
for i, arr in enumerate(tqdm(images)):
    stack_imgs[i] = arr

print("processing labels data.")
new_lbls = np.array(labels, dtype='int')

print("Stacking done. Statistics:-")
print("Image data shape: ", stack_imgs.shape)
print("Labels array shape: ", new_lbls.shape)

np.save(dir_path + img_file_name,stack_imgs)
np.save(dir_path + labels_file_name,new_lbls)