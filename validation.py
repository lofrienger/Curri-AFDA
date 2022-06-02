import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils


def validation_multi(args, model: nn.Module, criterion, valid_loader):
    with torch.no_grad():
        model.eval()
        losses = []
        confusion_matrix = np.zeros(
            (args.num_classes, args.num_classes), dtype=np.uint32)
        for inputs, targets in tqdm(valid_loader):
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, args.num_classes)

        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}
        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}
        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))

        print('Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.5f}'.format(
            valid_loss, average_iou, average_dices))
        metrics = {'valid_loss': valid_loss,
                   'iou': average_iou, 'dice': average_dices}
        metrics.update(ious)
        metrics.update(dices)
        return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

# https://www.programcreek.com/python/?CodeExample=calculate+iou
# https://calebrob.com/ml/2018/09/11/understanding-iou.html
def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious

# https://www.programcreek.com/python/?action=search_nlp&CodeExample=calculate_dice&submit=Search
def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices
