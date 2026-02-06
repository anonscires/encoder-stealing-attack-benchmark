import os
import sys
# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(sys.path)

import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import torchvision.transforms as transforms
from src.models.resnet import ResNetEncoder
from src.models.dino import DinoEncoder
from src.models.clip import CLIPVisionEncoder

def run_model(victim_model, x, arch):
    if 'clip' in arch:
        transform = CLIPVisionEncoder.get_transform()
        x = transform(x)
        x = victim_model(x)
    elif 'dino' in arch or 'vit' in arch:
        transform = DinoEncoder.get_transform()
        x = transform(x)
        x = victim_model(x)
    elif 'resnet' in arch:
        x = victim_model(x)
    else:
        raise ValueError(f"Unsupported victim architecture: {arch}")
    return x

def train_representation(target_encoder,surrogate_model,train_loader,criterion,optimizer,device, args):
    loss_epoch = 0
    mse_epoch = 0
    num_queries = 9000
    total_queries = 0

    # Load CLIP class embeddings for CIFAR-10 for text guided stealing
    embed_path = "" # TODO: Specify the correct path to the class embeddings file
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Class-embeds file not found: {embed_path}")
    class_embeds = torch.load(embed_path, map_location=device)

    if not torch.is_tensor(class_embeds):
        class_embeds = torch.tensor(class_embeds)
    
    class_embeds = class_embeds.to(device)
    print("Loaded class embeddings shape:", getattr(class_embeds, "shape", None))

    for step, (x, y) in enumerate(tqdm(train_loader, desc="Training Representation")):
        target_encoder.eval()

        surrogate_model.train()
        optimizer.zero_grad()
        target_encoder.requires_grad = False
        x = x.to(device)
        y = y.to(device)


        # print('x shape:', x.shape)
        # print('y shape:', y)

        # Build a (batch_size, embed_dim) tensor of class embeddings corresponding to y
        y_idx = y.long().squeeze()
        if y_idx.dim() == 0:
            y_idx = y_idx.unsqueeze(0)
        # Make sure indices are on the same device as class_embeds for indexing
        y_idx = y_idx.to(class_embeds.device)
        batch_class_embeds = class_embeds[y_idx]  # shape: (batch_size, embed_dim)
        # Ensure result is on the training device
        batch_class_embeds = batch_class_embeds.to(device)
        # print("Batch class embeddings shape:", getattr(batch_class_embeds, "shape", None))

        # print("classes", y)
        # print("class embeds", batch_class_embeds[:, 0])

        # print("assert", (y == batch_class_embeds[:,0]).float().mean())

        # raise Exception("Debugging: Stopping execution to inspect shapes.")
    
        re = run_model(target_encoder, x, args.victim_arch)
        su_output = run_model(surrogate_model, x, args.surrogate_arch)
        loss = criterion(su_output, re) + criterion(su_output, batch_class_embeds)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        mse = F.mse_loss(su_output,re)
        mse_epoch += mse.item()

        total_queries += len(x)
        if total_queries >= num_queries:
            break

    print("loss")
    print(loss_epoch/len(train_loader))
    print("mse")
    print(mse_epoch/len(train_loader))
    print("")

def train_represnetation_linear(model,target_encoder,train_loader,criterion,optimizer,device):
    accuracy_sample = []
    total_sample = []
    target_sample = []
    loss_epoch = 0
    for step, (x,y) in enumerate(train_loader):
        model.encoder.requires_grad = False
        model.encoder.eval()
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        model.linear.train()
        re = model.encoder(x)
        output = model.linear(re)
        loss = criterion(output, y)
        predicted = output.argmax(1)
        predicted = predicted.cpu().numpy()
        predicted = list(predicted)
        accuracy_sample.extend(predicted)
        y = y.cpu().numpy()
        y = list(y)
        total_sample.extend(y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        target_linear.eval()
        target_encoder.eval()
        t_output = target_linear(target_encoder(x))
        t_predicted = t_output.argmax(1)
        t_predicted = t_predicted.cpu().numpy()
        t_predicted = list(t_predicted)
        target_sample.extend(t_predicted)
    print("accuracy:")
    print(accuracy_score(total_sample, accuracy_sample))
    print("agreement:")
    print(accuracy_score(target_sample,accuracy_sample))
    print("loss")
    print(loss_epoch / len(train_loader))
