import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import torch.nn as nn
from scipy.special import rel_entr

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
GENDER_CLASSES = ['Male', 'Female']
RACE_CLASSES = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern', 'Other']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ResNet34 for FairFace
def resnet34(num_classes=18, pretrained=True):
    model = models.resnet34(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load model
def load_fairface_model(weight_path):
    model = resnet34(num_classes=18)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict for a single image
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        gender_pred = outputs[:, :2].argmax(dim=1).item()
        race_pred = outputs[:, 2:].argmax(dim=1).item()
    return GENDER_CLASSES[gender_pred], RACE_CLASSES[race_pred]

# Predict over a folder
def evaluate_folder(model, folder_path):
    gender_counts = {g: 0 for g in GENDER_CLASSES}
    race_counts = {r: 0 for r in RACE_CLASSES}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            gender, race = predict_image(model, os.path.join(folder_path, fname))
            gender_counts[gender] += 1
            race_counts[race] += 1
    return gender_counts, race_counts

# KL Divergence
def kl_divergence(pred_dist, ref_dist):
    p = np.array(pred_dist) / sum(pred_dist)
    q = np.array(ref_dist) / sum(ref_dist)
    return sum(rel_entr(p, q))

# Unified classification for a single image
def classify_demographics(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)[0]

    gender_logits = output[:2]
    race_logits = output[2:]

    gender = GENDER_CLASSES[torch.argmax(gender_logits).item()]
    race = RACE_CLASSES[torch.argmax(race_logits).item()]
    return gender, race
