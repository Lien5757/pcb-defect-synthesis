import os
import cv2
import random
import shutil

def change_filename_with_class(src_dir, dest_dir, defect_name, size=(512,512)):
    """Resize and rename images by class with sequential numbering"""
    classes = os.listdir(src_dir)
    
    idx=0
    for class_name in classes:
        dest_image_dir = os.path.join(dest_dir, 'images', class_name)
        os.makedirs(dest_image_dir, exist_ok=True)

        fileList = os.listdir(os.path.join(src_dir, class_name))
        for filename in fileList:
            image_path = os.path.join(src_dir, class_name, filename)

            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                new_image_path = os.path.join(dest_image_dir, f"{defect_name}_{idx:03d}.png")
                
                cv2.imwrite(new_image_path, image)
                print(f'{image_path} is saved to {new_image_path}')
                idx += 1
    
    print(f'Finished! Total is {idx}')

def resize_images(data_dir, size=(512, 512)):
    """Resize all images in dataset to specified dimensions"""
    image_dir=os.path.join(data_dir,'images')
    classes = os.listdir(image_dir)
    
    for class_name in classes:
        dest_image_dir = os.path.join(image_dir, class_name)
        os.makedirs(dest_image_dir, exist_ok=True)

        fileList = os.listdir(os.path.join(image_dir, class_name))
        for file_name in fileList:
            image_path = os.path.join(image_dir, class_name, file_name)

            # Check if the file is an image
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Read the image
                img = cv2.imread(image_path)
                filename = file_name.split('.')[0]

                if img is not None:
                    # Resize the image
                    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC) # cv2.INTER_AREA : 用於縮小影像

                    # Save the resized image to the output directory
                    output_path = os.path.join(dest_image_dir, f'{filename}.png')
                    cv2.imwrite(output_path, resized_img)
                    print(f"Resize : {image_path} -> {output_path}")
                else:
                    print(f"Could not read image: {image_path}")
            else:
                print(f"Skipped non-image file: {image_path}")

if __name__ == "__main__":
    # ============ CONFIG ============
    TASK = "resize"  # "resize" or "rename"

    # ============ Task 1: Resize (keep original filenames) ============
    if TASK == "resize":
        data_dir = r'datasets\train\exp1'
        print(f"Resizing images in: {data_dir}")
        resize_images(data_dir, size=(512, 512))

    # ============ Task 2: Resize + Rename ============
    elif TASK == "rename":
        src_dir = r''
        dest_dir = r'datasets\train\exp1'
        print(f"Resizing and renaming images from: {src_dir}")
        change_filename_with_class(src_dir=src_dir,
                                   dest_dir=dest_dir,
                                   defect_name='defect',
                                   size=(512, 512))

