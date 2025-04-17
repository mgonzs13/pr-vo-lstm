import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers)


# TODO: Load dataset
transform = T.Compose([T.ToTensor(), model.resnet_transforms()])

validation_dataloader = ...


# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt", weights_only=True))
model.eval()

validation_string = ""
position = [0.0] * 7

with torch.no_grad():
    for images, labels, timestamps in tqdm(validation_dataloader, f"Validating:"):

        images = images.to(device)
        labels = labels.to(device)

        target = model(images).cpu().numpy().tolist()[0]

        # TODO: add the results into the validation_string


f = open("validation.txt", "a")
f.write(validation_string)
f.close()
