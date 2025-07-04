import evaluate
import torch
from torch import autocast

accuracy = evaluate.load("accuracy")
mem_args = dict(memory_format=torch.channels_last)


def evaluate_accuracy(model, valid, device):
    model.eval()
    num_batches_validation = len(valid)
    acc_score = 0

    with autocast(device.type):
        for images, mask_validation in valid:
            # Send to correct device
            image = image.to(device, dtype=torch.float32)
            mask_validation = mask_validation.to(device, dtype=torch.float32)

            # Generate predicted mask
            mask_prediction = model(image)
            # Calculate accuracy score
            acc_score += accuracy.compute(mask_prediction, mask_validation)

    return acc_score / max(num_batches_validation, 1)
