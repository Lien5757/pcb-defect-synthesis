import os

def check_files_exist(data_dir, filename):
    """Check if image, mask, and text files exist for a given filename"""
    image_path = os.path.join(data_dir, "images", filename)
    mask_path = os.path.join(data_dir, "masks", filename)
    text_path = os.path.join(data_dir, "texts", filename)

    image_exists = os.path.exists(image_path)
    mask_exists = os.path.exists(mask_path)
    text_exists = os.path.exists(text_path)

    return image_exists and mask_exists and text_exists

def check_data_all_set(data_dir):
    """Verify that all images have corresponding mask and text files"""
    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        print(f"Error: images directory not found at {image_dir}")
        return

    missing_count = 0
    for img_filename in os.listdir(image_dir):
        if not check_files_exist(data_dir, img_filename):
            print(f"Missing: {img_filename} - mask or text file is missing")
            missing_count += 1

    if missing_count == 0:
        print(f"✓ All data is properly paired!")
    else:
        print(f"✗ Found {missing_count} missing files")

if __name__ == "__main__":
    # ============ CONFIG: Specify the dataset directory to check ============
    data_dir = r'datasets\train\exp1'

    print(f"Checking data integrity in: {data_dir}")
    check_data_all_set(data_dir=data_dir)
