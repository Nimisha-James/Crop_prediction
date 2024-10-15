from torchvision import models

import torch.nn as nn
import os

# OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def build_model(fine_tune=True, num_classes=10):
    model = models.swin_t(weights='DEFAULT')
    print(model)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    
    model.head = nn.Linear(
        in_features=768, 
        out_features=num_classes, 
        bias=True
    )
    return model

if __name__ == '__main__':
    model = build_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")