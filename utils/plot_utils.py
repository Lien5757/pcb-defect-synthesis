import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_live(epoch, loss_history, plot_interval=1, save_dir=''):
    # Update the plot every `plot_interval` epochs
    if (epoch + 1) % plot_interval == 0:
        plt.ion()  # Turn on interactive mode
        plt.figure("Training Loss", figsize=(6, 4))
        plt.clf()  # Clear the figure to prevent overlap
        plt.plot(range(1, len(loss_history) + 1), loss_history, linestyle='-', label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.draw()  # Draw the updated figure
        plt.pause(0.01)  # Pause briefly to allow the figure to update

    plt.savefig(os.path.join(save_dir, 'Loss.png'), format='png')

def plot_lr(lr_history, save_dir=''):
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("LR vs. Training Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_plot.png"))
    plt.show()

def show_batch_images_and_masks(batch_images, batch_masks, batch_mask_images, wait_time=500):
    """
    Display images and corresponding masks from a batch using OpenCV.

    Args:
        batch_images (torch.Tensor): Batch of images (B, C, H, W).
        batch_masks (torch.Tensor): Batch of masks (B, 1, H, W).
        wait_time (int): Time in milliseconds to display each image-mask pair.
    """
    # Ensure batch size matches
    assert batch_images.shape[0] == batch_masks.shape[0] == batch_mask_images.shape[0]

    batch_size = batch_images.shape[0]
    for i in range(batch_size):
        # Convert image to NumPy format
        image = batch_images[i].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        image = ((image + 1) * 127.5).astype(np.uint8)  # De-normalize to [0, 255]

        # Convert mask to NumPy format
        mask = batch_masks[i].squeeze(0).cpu().numpy()  # (1, H, W) -> (H, W)
        mask = (mask * 255).astype(np.uint8)  # Convert to binary mask (if normalized)

        # Convert mask image to NumPy format
        mask_image = batch_mask_images[i].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        mask_image = ((mask_image + 1) * 127.5).astype(np.uint8)  # De-normalize to [0, 255]

        # Convert RGB to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
        
        # Display image and mask side by side
        combined = np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), mask_image))  # Stack horizontally
        cv2.imshow("Image/mask/mask image", combined)

        # Wait and handle key press (optional)
        key = cv2.waitKey(wait_time)  # Wait for key input
        if key == 27:  # Press ESC to exit early
            cv2.destroyAllWindows()
