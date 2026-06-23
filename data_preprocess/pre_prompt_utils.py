import os
import shutil

def save_prompts(data_dir, class_to_prompt_map):
    """Generate and save prompts for all classes based on class_name mapping"""
    image_dir = os.path.join(data_dir, 'images')
    prompt_dir = os.path.join(data_dir, 'texts')

    if not os.path.exists(image_dir):
        print(f"Error: images directory not found at {image_dir}")
        return

    os.makedirs(prompt_dir, exist_ok=True)

    for class_name in os.listdir(image_dir):
        class_image_dir = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_image_dir):
            continue

        # Get prompt from mapping
        prompt = class_to_prompt_map.get(class_name, f"A {class_name} defect")

        class_prompt_dir = os.path.join(prompt_dir, class_name)
        os.makedirs(class_prompt_dir, exist_ok=True)

        for image_file in os.listdir(class_image_dir):
            # Create .txt file with same basename
            prompt_filename = f"{os.path.splitext(image_file)[0]}.txt"
            prompt_path = os.path.join(class_prompt_dir, prompt_filename)

            with open(prompt_path, "w") as f:
                f.write(prompt)

    print(f"✓ Prompts saved successfully to {prompt_dir}")

if __name__ == "__main__":
    # ============ PROMPT CONFIGURATION ============
    # Mapping: class_name -> prompt text
    CLASS_TO_PROMPT = {
        "0_crash": "A crash defect on tray",
        "1_scratch": "A scratch defect on tray",
        "2_white_contamination": "White contamination defect on tray",
        "3_red_contamination": "Red contamination defect on tray",
        "4_foreign_matter": "Foreign matter defect on tray"
    }

    # ============ CONFIG: Modify the dataset path ============
    data_dir = r'datasets\train\exp1'

    print(f"Generating prompts for: {data_dir}")
    save_prompts(data_dir, CLASS_TO_PROMPT)
