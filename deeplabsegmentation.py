import torch
import torchvision
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# path and file names
dir_path = './Data/'
img_file_name = 'train_images.npy'
labels_file_name = 'train_labels.npy'
output_folder = './Data/segmented/'


print("Loading our dataset...")
# looading our model data
X = np.load(dir_path + img_file_name, allow_pickle=True)
y = np.load(dir_path + labels_file_name, allow_pickle=True)


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((120, 160)),
    torchvision.transforms.ToTensor(),
])

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

predictions = []
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
ckpt_path = './ModelSaves/model_13_2_2_2_epoch_580.pth'
ckpt = torch.load(ckpt_path)

print(ckpt.keys())
exit(0)

model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)

model.eval()

print("Preprocessing and performing segmentation predictions...")
counter = 0
for image in tqdm(X):
    image = Image.fromarray(image)
    image = image.convert('RGB')

    input_tensor = preprocess(image).float().unsqueeze(0)
    input_tensor = normalize(input_tensor)

    input_batch = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # conver to np array
    output_predictions = output.argmax(0).cpu().numpy()
    predictions.append(output_predictions)
    counter +=1

    window = 1000

    if counter % window == 0:
        # Define the color palette for visualization
        class_labels = model.classifier[-1].weight.data.shape[0]
        color_palette = np.random.randint(0, 256, (class_labels, 3), dtype=np.uint8)
        color_palette = torch.tensor(color_palette).unsqueeze(0)

        for i, prediction in enumerate(tqdm(predictions)):
            output_path = output_folder + f'image_{counter//window}_{i}.png'
            prediction_visual = Image.fromarray(prediction.astype(np.uint8)).resize(image.size)
            prediction_visual.putpalette(color_palette.numpy())
            prediction_visual.save(output_path)

        predictions = []





# # saving segmented images
# print("Saving in folder `Data/segmented/`...")
# output_folder = './Data/segmented/'


