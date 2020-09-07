from inference.models.grconvnet_lightning import GraspModule
# from PIL import Image
from utils.dataset_processing.image import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import os
from utils.data import get_dataset

print(torch.__version__)

# ckpt = torch.load('trained-models/epoch152_cpu.ckpt', map_location=torch.device('cpu'))
# model = GraspModule.load_from_checkpoint('trained-models/epoch1_cpu.ckpt')
model = torch.load('trained-models/epoch83_cpu.pt')

# print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

# Load the RGB image
def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

def _get_crop_attrs(img, output_size=1000):
    center = (img.shape[0]//2, img.shape[1]//2)
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    return center, left, top

def get_rgb(file, rot=0, zoom=1.0, normalise=True, output_size=1000):
    rgb_img = Image.from_file(file)
    center, left, top = _get_crop_attrs(rgb_img)
    rgb_img.rotate(rot, center)
    rgb_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
    rgb_img.zoom(zoom)
    rgb_img.resize((224, 224))
    if normalise:
        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

filepath = 'data/wild/istockphoto-623280930-170667a.jpg'
rgb_img = get_rgb(filepath, 0, 1)
img = numpy_to_torch(rgb_img).unsqueeze(0)


# # Should be (1,n_channels,224,224)
# img = Image.from_file('data/wild/IMG_20200829_010106.jpg')
# # img = Image.from_file('data/cornell/04/pcd0400r.png')
# img.resize(shape=(224,224))
# img.normalise()
# img = np.moveaxis(img, 2, 0)
# img = np.expand_dims(img, 0)
# img = torch.from_numpy(img.astype(np.float32)/255)
#
model.eval()
y_hat = model(img)
pos_output, cos_output, sin_output, width_output = y_hat
q = pos_output[0].detach().numpy()
q = np.moveaxis(q, 0, 2)
w = width_output[0].detach().numpy()
w = np.moveaxis(w, 0, 2)
plt.subplot(1,3,1)
plt.imshow(get_rgb(filepath, normalise=False))
plt.subplot(1,3,2)
plt.imshow(q)
plt.subplot(1,3,3)
plt.imshow(w)
plt.show()