import os
from typing import Dict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompt_loader import get_class_to_prompt_map


def save_prompts(data_dir: str, class_to_prompt_map: Dict[str, str]) -> None:
    """Generate and save prompts for all classes based on class_name mapping.

    Args:
        data_dir: Root directory containing 'images' subdirectory.
        class_to_prompt_map: Dictionary mapping class names to prompt texts.
    """
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

        prompt = class_to_prompt_map.get(class_name, f"A {class_name} defect")

        class_prompt_dir = os.path.join(prompt_dir, class_name)
        os.makedirs(class_prompt_dir, exist_ok=True)

        for image_file in os.listdir(class_image_dir):
            prompt_filename = f"{os.path.splitext(image_file)[0]}.txt"
            prompt_path = os.path.join(class_prompt_dir, prompt_filename)

            with open(prompt_path, "w") as f:
                f.write(prompt)

    print(f"✓ Prompts saved successfully to {prompt_dir}")


if __name__ == "__main__":
    # ============ CONFIG: Modify dataset name and path ============
    data_name = 'exp1'  # Choose from: exp2, exp3, exp4, exp5, exp5_v2, exp6
    data_dir = r'datasets\train\exp1'

    # Load prompts from unified config
    try:
        class_to_prompt = get_class_to_prompt_map(data_name)
        print(f"Generating prompts for: {data_dir}")
        print(f"Dataset: {data_name}")
        save_prompts(data_dir, class_to_prompt)
    except ValueError as e:
        print(f"Error: {e}")
