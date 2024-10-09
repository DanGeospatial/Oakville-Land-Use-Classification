"""
Train the land use dataset with 3 classes and 4 image bands
Currently using RMSprop instead of adam optimizer
"""

# Import PyTorch libraries
import torch
import torch.nn as nn
from torch import device, cuda, optim, autocast, save
from torch.optim import Optimizer

# Import other libraries

# Import algorithms
from Models.Simple_CNN import Net
from utils.validation_accuracy import evaluate_accuracy
from Data.dataset_Oakville_v2 import train_set, validation_set, getLength

num_classes = 5
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.CrossEntropyLoss()
    gradient_scaler = torch.amp.GradScaler()

    step_glob = 0

    # Start training
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for batch in train_set:
            images = batch["image"]
            masks = batch["mask"]
            images = images.to(device=device_hw)
            masks = masks.to(device=device_hw)

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
    print("Using PyTorch version: ", torch.__version__)
    print("With CUDA version: ", torch.cuda_version)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    network = Net(num_bands, num_classes)
    network = network.to(device)

    try:
        train_model(
            model=network,
            device_hw=device,
            epoch_num=epochs,
            lr=learning_rate,
            wd=weight_decay,
            mom=momentum
        )
    except cuda.OutOfMemoryError:
        print("Out of memory!")
        cuda.empty_cache()
