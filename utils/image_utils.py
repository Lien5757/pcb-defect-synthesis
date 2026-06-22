from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import cv2

def resize_image(image, target_size=(512, 512)):
    if isinstance(image, Image.Image):  
        image = np.array(image)

    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    return Image.fromarray(resized_image)

def random_flip_rotate_pil(image):
    """
    Randomly flip (horizontal or vertical) and rotate (90, 180, or 270 degrees) a PIL image.
    """
    # Random flip
    flip_mode = random.choice(['H', 'V', None])
    if flip_mode == 'H':
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_mode == 'V':
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation
    rotation_angle = random.choice([90, 180, 270])
    image = image.rotate(rotation_angle, expand=True)

    return image

def combine_image_grid_batch(image_lists):
    rows = len(image_lists)
    cols = max(len(row) for row in image_lists)

    img_width, img_height = image_lists[0][0].size
    combined = Image.new('RGB', (cols * img_width, rows * img_height))

    for row_idx, row in enumerate(image_lists):
        for col_idx, img in enumerate(row):
            combined.paste(img, (col_idx * img_width, row_idx * img_height))

    return combined

def combine_image_with_prompt(base_img, mask_img, result_img, prompt, size=(256, 256)):
    base = base_img.resize(size)
    mask = mask_img.resize(size)
    result = result_img.resize(size)

    # 建立新畫布，包含上方文字空間
    width, height = size
    font_height = 24
    combined = Image.new('RGB', (width * 3, height + font_height), color=(255, 255, 255))

    # 貼上圖像
    combined.paste(base, (0, font_height))
    combined.paste(mask, (width, font_height))
    combined.paste(result, (width * 2, font_height))

    # 寫入 prompt 文字
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    draw.text((5, 0), prompt, fill=(0, 0, 0), font=font)
    return combined
