import os

def check_files_exist(data_dir, filename):
    # 資料夾路徑
    image_path = os.path.join(data_dir, "images", filename)
    mask_path = os.path.join(data_dir, "masks", filename)
    text_path = os.path.join(data_dir, "texts", filename)

    # 檢查檔案是否存在
    image_exists = os.path.exists(image_path)
    mask_exists = os.path.exists(mask_path)
    text_exists = os.path.exists(text_path)

    # 如果三個檔案都存在，回傳 True，否則回傳 False
    return image_exists and mask_exists and text_exists

def check_data_all_set(data_dir):
    imgList = os.listdir(os.path.join(data_dir, 'images'))
    
    for img_filename in imgList:
        isExist = check_files_exist(data_dir, img_filename)

        if isExist:
            continue
        else:
            print(f'Index {img_filename} of Mask or Text is not exist.')
    print('All the data is paired!')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify that all data (images, masks, prompts) is properly paired'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to data directory containing images/, masks/, and texts/ folders'
    )

    args = parser.parse_args()

    print(f"Checking data completeness in {args.data_dir}...")
    check_data_all_set(data_dir=args.data_dir)