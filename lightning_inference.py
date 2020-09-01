from inference.models.grconvnet_lightning import GenerativeResnet
from PIL import Image
import numpy as np
import torch

print(torch.__version__)

# ckpt = torch.load('trained-models/epoch152_cpu.ckpt', map_location=torch.device('cpu'))
model = GenerativeResnet(args=None, input_channels=3).load_from_checkpoint('trained-models/epoch152_cpu.ckpt')

# print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

x = torch.Tensor(np.array(Image.open('data/wild/IMG_20200829_010106.jpg'))).unsqueeze(0)

model.eval()
y_hat = model(x)