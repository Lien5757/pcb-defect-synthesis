import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_unique_save_path(output_dir, base_name="comparison", ext=".png"):
    os.makedirs(output_dir, exist_ok=True)
    idx = 1
    while True:
        save_path = os.path.join(output_dir, f"{base_name}_{idx}{ext}")
        if not os.path.exists(save_path):
            return save_path
        idx += 1

def compare_base_gen(base_path, generated_path, output_dir=''):
    base_image = cv2.imread(base_path)
    generated_image = cv2.imread(generated_path)

    height, width = generated_image.shape[:2]
    base_image = cv2.resize(base_image, (width, height))

    # Method 1: Absolute difference
    base_hsv = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)
    gen_hsv = cv2.cvtColor(generated_image, cv2.COLOR_BGR2HSV)
    hsv_diff = cv2.absdiff(gen_hsv, base_hsv)
    # amplified_diff = cv2.convertScaleAbs(hsv_diff, alpha=1)  # Amplify differences ×5

    # Method 2: SSIM heatmap
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
    score, ssim_diff = ssim(gray_base, gray_gen, full=True)
    ssim_diff = (ssim_diff * 255).astype("uint8")
    ssim_diff_colored = cv2.applyColorMap(ssim_diff, cv2.COLORMAP_JET)

    # Save result
    save_path = get_unique_save_path(output_dir)

    comb = np.hstack((base_image, generated_image, hsv_diff, ssim_diff_colored))
    cv2.putText(comb, f'SSIM: {score:.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, comb)
    cv2.imshow(f'SSIM: {score:.4f}', comb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_base_gen(base_path=r"C:\Users\Owner\Desktop\Lien-SD_Inpaint\datasets\test\clean_images(689)\ka412195a03_2INT_a210219_6-3ds_342_9_ROI0_OPOK.png",
                     generated_path=r"C:\Users\Owner\Desktop\Lien-SD_Inpaint\output\exp2\base_image(10)\exp2_b=4_256x256_DDIM\inpaint_2_5.png",
                     output_dir=r'output\exp2\comparison')