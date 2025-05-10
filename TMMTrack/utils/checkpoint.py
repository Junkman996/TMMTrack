import torch
import os

def save_model(state, path):
    torch.save(state, path)

def load_model(model, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    return checkpoint