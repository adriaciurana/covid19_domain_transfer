import os
import sys

import torch
import torch.optim as optim
from torch import nn
from logger import Logger
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_creator import ModelCreator
from dataloaders.covid19dataset import *

ARCH = 'resnet34'
EPOCHS = 300
CHECKPOINT_SAVE_INTERVAL = 15
MIN_DISCRIMINATOR_EPOCH = 5
IMAGES_TRAIN_FOLDER = "/media/Datos/git/coronavirus/radiology_detector/data_test/labeled"
IMAGES_TRAIN_UNLABELED = "/media/Datos/git/coronavirus/radiology_detector/data_test/unlabeled"
IMAGES_TEST_FOLDER = "/media/Datos/git/coronavirus/radiology_detector/data_test/labeled"
IMAGES_TEST_UNLABELED = "/media/Datos/git/coronavirus/radiology_detector/data_test/unlabeled"

BATCH_SIZE_LABELED = 6
BATCH_SIZE_UNLABELED = 6
NUM_WORKERS = 6

MEAN = np.array([127, 127, 127], dtype=np.float32)
STD = np.array([127, 127, 127], dtype=np.float32)
PAD = 32
SIZE = [224*2, 224*2]
DEVICE = torch.device("cuda:1")

LOGGER_PATTERN = {
    'TRAIN': "[TRAIN] Epoch {epoch:d} | {losses:s} | {metrics:s}",
    'VAL': "[VAL] Epoch {epoch:d} | {losses:s} | {metrics:s}"
}
MODEL_PATTERN = "checkpoint_{epoch:d}_{loss:f}.pkl"
CHECKPOINT_FOLDER = "checkpoints"
OUTPUT_LOGS = "logs"
os.makedirs(OUTPUT_LOGS, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# Get dataloader
train_dataset = Covid19Dataset(IMAGES_TRAIN_FOLDER, 
    do_aug='train', 
    mean=MEAN,
    std=STD, 
    pad=PAD,
    size=SIZE)

train_unlabeled_dataset = Covid19DatasetUnlabeled(IMAGES_TRAIN_UNLABELED, 
    do_aug='train', 
    mean=MEAN,
    std=STD, 
    pad=PAD,
    size=SIZE)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_LABELED,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #collate_fn=train_dataset.collate_fn()
)

train_unlabeled_dataloader = InfiniteDataLoader(
    train_unlabeled_dataset,
    batch_size=BATCH_SIZE_UNLABELED,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #collate_fn=train_unlabeled_dataset.collate_fn()
)



test_dataset = Covid19Dataset(IMAGES_TEST_FOLDER, 
    do_aug='test', 
    mean=MEAN,
    std=STD, 
    pad=PAD,
    size=SIZE)

test_unlabeled_dataset = Covid19DatasetUnlabeled(IMAGES_TEST_UNLABELED, 
    do_aug='test', 
    mean=MEAN,
    std=STD, 
    pad=PAD,
    size=SIZE)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_LABELED,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #collate_fn=test_dataset.collate_fn()
)

test_unlabeled_dataloader = InfiniteDataLoader(
    test_unlabeled_dataset,
    batch_size=BATCH_SIZE_UNLABELED,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #collate_fn=test_unlabeled_dataset.collate_fn()
)

# Get model
backbone, classifier, discriminator = ModelCreator(
    inference=False, 
    arch=ARCH)
backbone = backbone.to(DEVICE)
classifier = classifier.to(DEVICE)
discriminator = discriminator.to(DEVICE)

# optimizers
optimizer_generator = torch.optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()))
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters())

# Writters
logger = Logger(
    LOGGER_PATTERN,
    OUTPUT_LOGS
)

# Losses
criterion_generator = nn.BCELoss()
criterion_discriminator = nn.BCELoss()

# Metrics
def accuracy_metric(y_pred, y_true):
    return torch.mean((torch.round(y_pred) == y_true).type(torch.float))

def precision_metric(y_pred, y_true):
    cmp_ = (torch.round(y_pred) == y_true).type(torch.bool)
    tp = torch.sum((cmp_ & (y_true == 1).type(torch.bool)).type(torch.float))
    fp = torch.sum((~cmp_ & (y_true == 0).type(torch.bool)).type(torch.float))

    return tp / (tp + fp + 1e-10)

def recall_metric(y_pred, y_true):
    cmp_ = (torch.round(y_pred) == y_true).type(torch.bool)
    tp = torch.sum((cmp_ & (y_true == 1).type(torch.bool)).type(torch.float))
    fn = torch.sum((~cmp_ & (y_true == 1).type(torch.bool)).type(torch.float))

    return tp / (tp + fn + 1e-10)

def train(epoch, dataloader, unlabeled_dataloader):
    logger.clear()
    
    backbone.train()
    classifier.train()
    discriminator.train()
    
    pbar = tqdm(zip(dataloader, unlabeled_dataloader), total=len(dataloader))
    for batch_i, ((images, y_labels), images_unlabeled) in enumerate(pbar):
        losses = {}
        metrics = {}
    
        images = images.to(DEVICE)
        y_labels = y_labels.to(DEVICE)
        images_unlabeled = images_unlabeled.to(DEVICE)

        features = backbone(images)
        #features_unlabeled = backbone(images_unlabeled)

        """
            TRAIN DISCRIMINATOR
        """
        if epoch > MIN_DISCRIMINATOR_EPOCH:
            discriminator.train()
            optimizer_discriminator.zero_grad()

            #features = backbone(images)
            features_unlabeled = backbone(images_unlabeled)

            y_disc = discriminator(features.detach())
            y_disc_unlabeled = discriminator(features_unlabeled.detach())
            y_disc_comb = torch.cat((y_disc, y_disc_unlabeled), dim=0)

            # Loss for discriminator
            y_labels_disc = torch.cat((torch.ones(images.size(0), dtype=images.dtype, device=images.device), torch.zeros(images_unlabeled.size(0), dtype=images.dtype, device=images.device)), dim=0)
            loss_disc = criterion_discriminator(y_disc_comb, y_labels_disc)
            losses['discriminator'] = loss_disc.detach().cpu().numpy()

            accuracy_value = accuracy_metric(y_disc_comb.view(-1), y_labels_disc)
            precision_value = precision_metric(y_disc_comb.view(-1), y_labels_disc)
            recall_value = recall_metric(y_disc_comb.view(-1), y_labels_disc)
            metrics['accuracy_discriminator'] = float(accuracy_value.detach().cpu().numpy())
            metrics['precision_discriminator'] = float(precision_value.detach().cpu().numpy())
            metrics['recall_discriminator'] = float(recall_value.detach().cpu().numpy())

            # Update loss
            loss_disc.backward()
            optimizer_discriminator.step()

        """
            TRAIN GENERATOR
        """
        discriminator.eval()
        optimizer_generator.zero_grad()

        if epoch > MIN_DISCRIMINATOR_EPOCH:
            #y_disc = discriminator(features)
            #y_disc_unlabeled = discriminator(features_unlabeled)
            y_disc_comb = y_disc_comb.detach() #torch.cat((y_disc, y_disc_unlabeled), dim=0)

        y_class = classifier(features)

        # Loss for generator
        if epoch > MIN_DISCRIMINATOR_EPOCH:
            y_labels_disc = torch.cat((torch.ones(images.size(0), dtype=images.dtype, device=images.device), torch.ones(images_unlabeled.size(0), dtype=images.dtype, device=images.device)), dim=0)
        loss_gen = criterion_generator(y_class.view(-1), y_labels)

        if epoch > MIN_DISCRIMINATOR_EPOCH:
            loss_gen += criterion_discriminator(y_disc_comb, y_labels_disc)
        losses['generator'] = float(loss_gen.detach().cpu().numpy())

        # Metrics
        accuracy_value = accuracy_metric(y_class.view(-1), y_labels)
        precision_value = precision_metric(y_class.view(-1), y_labels)
        recall_value = recall_metric(y_class.view(-1), y_labels)
        metrics['accuracy_generator'] = float(accuracy_value.detach().cpu().numpy())
        metrics['precision_generator'] = float(precision_value.detach().cpu().numpy())
        metrics['recall_generator'] = float(recall_value.detach().cpu().numpy())

        loss_gen.backward()
        optimizer_generator.step()

        logger.show_bar_train(pbar, epoch, losses, metrics=metrics)

    # Save each epoch
    logger.write_train_epoch(pbar, epoch)

def test(epoch, dataloader, unlabeled_dataloader):
    logger.clear()
    
    backbone.eval()
    classifier.eval()
    discriminator.eval()
    
    with torch.no_grad():
        pbar = tqdm(zip(dataloader, unlabeled_dataloader), total=len(dataloader))
        for batch_i, ((images, y_labels), images_unlabeled) in enumerate(pbar):
            losses = {}
            metrics = {}
        
            images = images.to(DEVICE)
            y_labels = y_labels.to(DEVICE)
            images_unlabeled = images_unlabeled.to(DEVICE)

            features = backbone(images)
            #features_unlabeled = backbone(images_unlabeled)

            """
                TEST DISCRIMINATOR
            """
            if epoch > MIN_DISCRIMINATOR_EPOCH:
                #features = backbone(images)
                features_unlabeled = backbone(images_unlabeled)

                y_disc = discriminator(features.detach())
                y_disc_unlabeled = discriminator(features_unlabeled.detach())
                y_disc_comb = torch.cat((y_disc, y_disc_unlabeled), dim=0)

                # Loss for discriminator
                y_labels_disc = torch.cat((torch.ones(images.size(0), dtype=images.dtype, device=images.device), torch.zeros(images_unlabeled.size(0), dtype=images.dtype, device=images.device)), dim=0)
                loss_disc = criterion_discriminator(y_disc_comb, y_labels_disc)
                losses['discriminator'] = loss_disc.detach().cpu().numpy()

                accuracy_value = accuracy_metric(y_disc_comb.view(-1), y_labels_disc)
                precision_value = precision_metric(y_disc_comb.view(-1), y_labels_disc)
                recall_value = recall_metric(y_disc_comb.view(-1), y_labels_disc)
                metrics['accuracy_discriminator'] = float(accuracy_value.detach().cpu().numpy())
                metrics['precision_discriminator'] = float(precision_value.detach().cpu().numpy())
                metrics['recall_discriminator'] = float(recall_value.detach().cpu().numpy())

            """
                TEST GENERATOR
            """
            if epoch > MIN_DISCRIMINATOR_EPOCH:
                #y_disc = discriminator(features)
                #y_disc_unlabeled = discriminator(features_unlabeled)
                y_disc_comb = y_disc_comb.detach() #torch.cat((y_disc, y_disc_unlabeled), dim=0)

            y_class = classifier(features)

            # Loss for generator
            if epoch > MIN_DISCRIMINATOR_EPOCH:
                y_labels_disc = torch.cat((torch.ones(images.size(0), dtype=images.dtype, device=images.device), torch.ones(images_unlabeled.size(0), dtype=images.dtype, device=images.device)), dim=0)
            
            loss_gen = criterion_generator(y_class.view(-1), y_labels)
            if epoch > MIN_DISCRIMINATOR_EPOCH:
                loss_gen += criterion_discriminator(y_disc_comb, y_labels_disc) 
            
            losses['generator'] = float(loss_gen.detach().cpu().numpy())

            # Metrics
            accuracy_value = accuracy_metric(y_class.view(-1), y_labels)
            precision_value = precision_metric(y_class.view(-1), y_labels)
            recall_value = recall_metric(y_class.view(-1), y_labels)
            metrics['accuracy_generator'] = float(accuracy_value.detach().cpu().numpy())
            metrics['precision_generator'] = float(precision_value.detach().cpu().numpy())
            metrics['recall_generator'] = float(recall_value.detach().cpu().numpy())

            if 'discriminator' in losses.keys():
                losses['loss'] = losses['generator'] + losses['discriminator']
            else:
                losses['loss'] = losses['generator']

            logger.show_bar_test(pbar, epoch, losses, metrics=metrics)

    # Save each epoch
    logger.write_test_epoch(pbar, epoch)

    return sum([v.avg_deque for v in logger.losses_avg['VAL'].values()])

import gc
for epoch in range(EPOCHS):
    gc.collect()
    train(epoch, train_dataloader, train_unlabeled_dataloader)
    gc.collect()
    loss = test(epoch, test_dataloader, test_unlabeled_dataloader)

    if epoch > 0 and epoch % CHECKPOINT_SAVE_INTERVAL == 0:
        print('Saved checkpoint')

        model_dict = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'classifier': classifier.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_generator': optimizer_generator.state_dict(),
            'optimizer_discriminator': optimizer_discriminator.state_dict()
        }

        torch.save(model_dict, os.path.join(CHECKPOINT_FOLDER, MODEL_PATTERN.format(loss=loss, epoch=epoch)))
