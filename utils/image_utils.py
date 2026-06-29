from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import Tuple, Union

def resize_image(image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    if isinstance(image, Image.Image):
        image = np.array(image)

    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    return Image.fromarray(resized_image)

def combine_image_grid_batch(image_lists: list[list[Image.Image]]) -> Image.Image:
    rows = len(image_lists)
    cols = max(len(row) for row in image_lists)

    img_width, img_height = image_lists[0][0].size
    combined = Image.new('RGB', (cols * img_width, rows * img_height))

    for row_idx, row in enumerate(image_lists):
        for col_idx, img in enumerate(row):
            combined.paste(img, (col_idx * img_width, row_idx * img_height))

    return combined

def combine_image_with_prompt(
    base_img: Image.Image,
    mask_img: Image.Image,
    result_img: Image.Image,
    prompt: str,
    size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    base = base_img.resize(size)
    mask = mask_img.resize(size)
    result = result_img.resize(size)

    width, height = size
    font_height = 24
    combined = Image.new('RGB', (width * 3, height + font_height), color=(255, 255, 255))

    combined.paste(base, (0, font_height))
    combined.paste(mask, (width, font_height))
    combined.paste(result, (width * 2, font_height))

    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    draw.text((5, 0), prompt, fill=(0, 0, 0), font=font)
    return combined
