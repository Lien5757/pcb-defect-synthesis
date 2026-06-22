import os
import cv2
import shutil
import numpy as np

def extract_and_filter_by_color(image_dir, mask_dir, output_dir, color_name, color_ranges):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'masks'))

    # Get HSV range for the specified color
    if color_name not in color_ranges:
        raise ValueError(f"Color '{color_name}' not in predefined ranges.")
    lower_bound, upper_bound = color_ranges[color_name]

    # List all images and masks
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    for image_file in image_files:
        # Match mask file with the same name
        if image_file in mask_files:
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)

            # Load image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Skipping {image_file}, failed to load.")
                continue

            # Create a binary mask where the region is '1' where mask > 0
            binary_mask = mask > 0

            # Extract the masked region in the image
            extracted_region = np.zeros_like(image)
            extracted_region[binary_mask] = image[binary_mask] # (H, W, 3)
            # cv2.imshow('extracted_region',  extracted_region)
            # cv2.waitKey(0)

            # Convert extracted region to HSV
            hsv_region = cv2.cvtColor(extracted_region, cv2.COLOR_BGR2HSV)

            # Create a mask for the specified HSV color range
            color_mask = cv2.inRange(hsv_region, np.array(lower_bound), np.array(upper_bound))

            # Check if any pixels match the color range
            if cv2.countNonZero(color_mask) > 0:
                # Copy the original image and mask to output directory
                output_image_path = os.path.join(output_dir, 'images', image_file)
                output_mask_path = os.path.join(output_dir, 'masks', image_file)
                shutil.copy(image_path, output_image_path)
                shutil.copy(mask_path, output_mask_path)
                print(f"Copied {image_file} to output directory (matches '{color_name}').")

def filter_masks_by_color(image_dir, mask_dir, output_dir, color_name, color_ranges):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'masks'))

    # Get HSV range for the specified color
    if color_name not in color_ranges:
        raise ValueError(f"Color '{color_name}' not in predefined ranges.")
    lower_bound, upper_bound = color_ranges[color_name]

    # List all images and masks
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    for mask_file in mask_files:
        if mask_file in image_files:  # Ensure matching image exists
            image_path = os.path.join(image_dir, mask_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Load mask
            mask = cv2.imread(mask_path)
            if mask is None:
                print(f"Skipping {mask_file}, failed to load mask.")
                continue

            # Convert mask to HSV
            mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

            # Check if the mask contains pixels in the specified color range
            color_mask = cv2.inRange(mask_hsv, np.array(lower_bound), np.array(upper_bound))

            # If mask has pixels in the color range, save image and mask
            if cv2.countNonZero(color_mask) > 0:
                output_image_path = os.path.join(output_dir, 'images', mask_file)
                output_mask_path = os.path.join(output_dir, 'masks', mask_file)
                shutil.copy(image_path, output_image_path)
                shutil.copy(mask_path, output_mask_path)
                print(f"Copied {mask_file} to output directory (matches '{color_name}').")

if __name__ == "__main__":
    # Input directories
    image_dir = r"datasets\AOI__dry_films\images"
    mask_dir = r"datasets\AOI__dry_films\Masks"
    output_dir = r"datasets\AOI__dry_films(blue)"

    # Color ranges
    color_ranges = {
        'black': ((0, 0, 0), (180, 255, 46)),
        'gray': ((0, 0, 46), (180, 43, 220)),
        'white': ((0, 0, 221), (180, 30, 255)),
        'red1': ((0, 43, 46), (10, 255, 255)),
        'red2': ((156, 43, 46), (180, 255, 255)),
        'orange': ((11, 43, 46), (25, 255, 255)),
        'yellow': ((26, 43, 46), (34, 255, 255)),
        'green': ((35, 43, 46), (77, 255, 255)),
        'blue': ((78, 43, 46), (124, 255, 255)),
        'purple': ((125, 43, 46), (155, 255, 255))
    }

    # Filter images based on color
    selected_color = 'blue'
    extract_and_filter_by_color(image_dir, mask_dir, output_dir, selected_color, color_ranges)