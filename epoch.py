"""
This function trains or evaluates a machine learning model for one epoch using a given dataset (represented by the dataloader) 
and returns the average loss for the epoch and the current learning rate.
"""

from config import *

def run_epoch(dataloader, model, optimizer, criterion, scheduler, is_training=False):
    """
    This function trains or evaluates a machine learning model for one epoch using a given dataset (represented by the dataloader) 
    and returns the average loss for the epoch and the current learning rate.

    Args:
        dataloader: An iterable object that provides batches of data for training or evaluation.
        model: The machine learning model to be trained or evaluated.
        optimizer: An optimizer object that updates the model's weights during training.
        criterion: A loss function that calculates the difference between the model's predictions and the ground truth labels.
        scheduler (optional): A learning rate scheduler that adjusts the learning rate during training.
        is_training (optional, default=False): A boolean flag indicating whether to train (True) or evaluate (False) the model.

    Returns:
        decimal: epoch_loss
        decimal: lr
    """
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr