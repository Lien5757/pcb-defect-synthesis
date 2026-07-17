import os
import cv2
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss_live(epoch, loss_history, plot_interval=1, save_dir=''):
    if (epoch + 1) % plot_interval == 0:
        plt.figure("Training Loss", figsize=(6, 4))
        plt.clf()
        plt.plot(range(1, len(loss_history) + 1), loss_history, linestyle='-', label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'Loss.png'), format='png')
    plt.close()

def plot_lr(lr_history, save_dir=''):
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("LR vs. Training Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_plot.png"))
    plt.close()

def show_batch_images_and_masks(batch_images, batch_masks, batch_mask_images, wait_time=500):
    """
    Display images and corresponding masks from a batch using OpenCV.
    (GUI display disabled in Colab; for server environments use master branch)

    Args:
        batch_images (torch.Tensor): Batch of images (B, C, H, W).
        batch_masks (torch.Tensor): Batch of masks (B, 1, H, W).
        wait_time (int): Unused in headless mode.
    """
    assert batch_images.shape[0] == batch_masks.shape[0] == batch_mask_images.shape[0]
