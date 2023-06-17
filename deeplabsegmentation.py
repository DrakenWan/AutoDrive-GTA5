import torch
import torchvision
import numpy as np

model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')

# Set the model to evaluation mode
model.eval()

# path and file names
dir_path = './Data/'
img_file_name = 'train_images.npy'
labels_file_name = 'train_labels.npy'

print("Loading our dataset...")
# looading our model data
X = np.load(dir_path + img_file_name, allow_pickle=True)
y = np.load(dir_path + labels_file_name, allow_pickle=True)


X = X.astype(float)/255 # normalization for training


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Preprocessing...")
input_tensor = preprocess(X)

print(input_tensor.shape)