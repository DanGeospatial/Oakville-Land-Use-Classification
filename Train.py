"""
Train the land use dataset with 3 classes and 4 image bands
"""

# Import PyTorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn
import torchvision.transforms as tf
from torch import device, cuda, optim, autocast, save
from torch.optim import Optimizer


# Import other libraries
from sklearn.metrics import classification_report

# Import algorithms
from Models.Simple_CNN import Net
from utils.validation_accuracy import evaluate_accuracy
from utils.dataset_Oakville_V1 import train_set, test_set, validation_set, getLength

num_classes = 3
num_bands = 4
epochs = 40
learning_rate = 1e-5
weight_decay = 1e-8
momentum = 0.999
mem_args = dict(memory_format=torch.channels_last)
out_path = "I:/LandUseClassification.pth"


def train_model(model, device_hw, epoch_num, lr, wd, mom):

    # Set the optimizer and learning rate scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = nn.CrossEntropyLoss()
    gradient_scaler = torch.cuda.amp.GradScaler()

    step_glob = 0

    # Start training
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for batch in train_set:
            images, masks = batch['image'], batch['mask']

            images = images.to(device=device_hw, **mem_args)
            masks = masks.to(device=device_hw, **mem_args)

            with autocast(device_hw.type):
                mask_prediction = model(images)
                loss = criterion(masks, mask_prediction)

            Optimizer.zero_grad(set_to_none=True)
            gradient_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            step_glob += 1
            epoch_loss += loss.item()
            print("Train loss: ", loss.item(), "Step: ", step_glob, "epoch: ", epoch)

            # Evaluate Model
            # 5 * batch size
            step_div = (getLength() // (5 * 10))
            if step_div > 0:
                if step_glob % step_div == 0:
                    validation_score = evaluate_accuracy(model, validation_set, device_hw)
                    scheduler.step(validation_score)

                    print("Validation Score: ", validation_score)

    print("Training Complete!")
    state_dict = model.state_dict()
    save(state_dict, out_path)
    print("Model Saved")


if __name__ == '__main__':
    print("Using PyTorch version: ", torch.torch_version)
    print("With CUDA version: ", torch.cuda_version)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    network = Net(num_bands, num_classes)
    network = network.to(device, **mem_args)

    try:
        train_model(
            model=network,
            epoch_num=epochs,
            lr=learning_rate,
            wd=weight_decay,
            mom=momentum
        )
    except cuda.OutOfMemoryError:
        print("Out of memory!")
        cuda.empty_cache()
