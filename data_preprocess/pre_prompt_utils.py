import os
import re
import shutil
from PIL import Image

def save_prompts_dry_film(data_dir):
    image_dir=os.path.join(data_dir,'images')
    prompt_dir=os.path.join(data_dir,'texts')
    os.makedirs(prompt_dir, exist_ok=True)

    for folder_name in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Extract base color and shade index from folder_name like 'purple_0'
        match_folder = re.match(r'(\w+)_(\w+)', folder_name)
        if not match_folder:
            prompt = f"A {folder_name} dry film residuel defect"
        else:
            shade, base_color = match_folder.group(1), match_folder.group(2)
            prompt = f"A {shade} {base_color} dry film residuel defect"

        for filename in os.listdir(folder_path):
                full_prompt_dir = os.path.join(prompt_dir, folder_name)
                os.makedirs(full_prompt_dir, exist_ok=True)

                output_path = os.path.join(full_prompt_dir, f"{filename.split('.')[0]}.txt")
                with open(output_path, "w") as f:
                    f.write(prompt)

    print("Prompts saved successfully.")

def save_prompts_scratch(data_dir):
    image_dir=os.path.join(data_dir,'images')
    prompt_dir=os.path.join(data_dir,'texts')
    os.makedirs(prompt_dir, exist_ok=True)

    for folder_name in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        if folder_name == "large_area":
            prompt = "A large scratch defect covering a wide area"
        elif folder_name == "single_line":
            prompt = "A thin single-line scratch defect on the PCB surface, narrow and sharp"
        elif folder_name == "muti_parallel_lines":
            prompt = "Multiple parallel scratch defects, aligned in one direction"
        elif folder_name == "wide_line_expose_base":
            prompt = "A wide linear scratch defect with base material exposed"

        for filename in os.listdir(folder_path):
            match_file = re.search(r'scratch_defect_pcb_(\d+)\.png', filename)
            if match_file:
                idx = match_file.group(1)

                full_prompt_dir = os.path.join(prompt_dir, folder_name)
                os.makedirs(full_prompt_dir, exist_ok=True)

                output_path = os.path.join(full_prompt_dir, f"defect_{idx}.txt")
                with open(output_path, "w") as f:
                    f.write(prompt)

    print("Prompts saved successfully.")

def save_prompts_tray(data_dir):
    image_dir=os.path.join(data_dir,'images')
    prompt_dir=os.path.join(data_dir,'texts')
    os.makedirs(prompt_dir, exist_ok=True)

    for folder_name in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        if folder_name == "0_crash":
            prompt = "A crash defect on tray"
        elif folder_name == "1_scratch":
            prompt = "A scratch defect on tray"
        elif folder_name == "2_white_contamination":
            prompt = "White contamination defect on tray"
        elif folder_name == "3_red_contamination":
            prompt = "Red contamination defect on tray"
        elif folder_name == "4_foreign_matter":
            prompt = "foreign matter defect on tray"

        for image_file in os.listdir(folder_path):
            filename = image_file.split('.')[0]

            full_prompt_dir = os.path.join(prompt_dir, folder_name)
            os.makedirs(full_prompt_dir, exist_ok=True)

            output_path = os.path.join(full_prompt_dir, f"{filename}.txt")
            with open(output_path, "w") as f:
                f.write(prompt)

    print("Prompts saved successfully.")

def copy_all_files(data_dir, target_dir):
    subfolders = ['images', 'masks', 'texts']
    os.makedirs(target_dir, exist_ok=True)

    for subfolder in subfolders:
        source_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} does not exist.")
            continue

        for class_folder in os.listdir(source_path):
            class_folder_path = os.path.join(source_path, class_folder)
            if not os.path.isdir(class_folder_path):
                continue

            for filename in os.listdir(class_folder_path):
                src_file = os.path.join(class_folder_path, filename)

                dst_file = os.path.join(target_dir, filename)
                shutil.copy2(src_file, dst_file)

    print(f"All files copied to {target_dir} successfully.")

if __name__ == "__main__":   
    ## 3、 Set prompt ==================================================================================
    # save_prompts_scratch(data_dir=r'datasets\AOI_06_sratch(512x512))

    # save_prompts_tray(data_dir=r'D:\Tray_datasets\Tray_exp1')

    save_prompts_dry_film(data_dir=r'datasets\AOI_dry_films\AOI__dry_films(all)_prompt_exp5_v2(100)_9')