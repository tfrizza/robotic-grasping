from inference.models.grconvnet_lightning import GraspModule
from utils.dataset_processing.image import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import os

print(torch.__version__)

# Load the RGB image
def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

def _get_crop_attrs(img, output_size=1000):
    center = (img.shape[0] // 2, img.shape[1] // 2)
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

def predict_grasp(img_path, model_path='trained-models/epoch83_cpu.pt', plot=False):
    '''
    Takes in an arbitrary image, preprocesses it (subject to change) to normalised 224x224x3
    and predicts using saved model
    --
    @img_path: path to img to predict on
    @model_path: path to saved model
    @plot: flag to plot the resulting grasp images
    --
    >returns: 3-tuple of (quality,width,angle) images, each of size (224x224x1)
    '''
    # ckpt = torch.load('trained-models/epoch152_cpu.ckpt', map_location=torch.device('cpu'))
    model = torch.load('trained-models/epoch83_cpu.pt')

    # img_path = 'data/wild/istockphoto-623280930-170667a.jpg'
    rgb_img = get_rgb(img_path, 0, 1) # This is all hard-coded for now to suit the trained model
    img = numpy_to_torch(rgb_img).unsqueeze(0)

    model.eval()
    y_hat = model(img)
    pos_output, cos_output, sin_output, width_output = y_hat
    # will do angle preds later
    q = pos_output[0].detach().numpy()
    q = np.moveaxis(q, 0, 2)
    w = width_output[0].detach().numpy()
    w = np.moveaxis(w, 0, 2)
    if plot:
        plt.subplot(1,3,1)
        plt.imshow(get_rgb(img_path, normalise=False))
        plt.title('Original')
        plt.subplot(1,3,2)
        plt.imshow(q)
        plt.title('Quality')
        plt.subplot(1,3,3)
        plt.imshow(w)
        plt.title('Width')
        plt.show()

    return q,w

if __name__ == '__main__':
    img_path = 'data/wild/istockphoto-623280930-170667a.jpg'
    model_path = 'trained-models/epoch83_cpu.pt'
    predict_grasp(img_path, model_path, plot=True)