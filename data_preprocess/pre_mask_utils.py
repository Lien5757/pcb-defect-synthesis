import os
import sys
import cv2
import numpy as np
import shutil

def draw_np(image_dir, mask_dir, image_index=0, isResize=True):
    os.makedirs(mask_dir, exist_ok=True)

    # 取得所有影像檔案
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    

    if not image_files:
        print("No images found in the source directory.")
        return

    # 筆刷設定
    brush_sizes = [20, 30, 50]
    brush_index = 0
    drawing = False
    points = []
    undo_stack = []

    def mouse_event(event, x, y, flags, param):
        nonlocal drawing, points, canvas, mask, undo_stack

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [[x, y]]
            # 儲存當前狀態（深拷貝）
            undo_stack.append((canvas.copy(), mask.copy()))

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            points.append([x, y])
            if len(points) >= 2:
                cv2.line(canvas, tuple(points[-2]), tuple(points[-1]), (255, 255, 255, 255), brush_sizes[brush_index])
                cv2.line(mask, tuple(points[-2]), tuple(points[-1]), (255, 255, 255), brush_sizes[brush_index])
            cv2.imshow('Drawing Canvas', canvas)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            points = []

    while image_index < len(image_files):
        # 讀取影像
        image_path = os.path.join(image_dir, image_files[image_index])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if isResize:
            image = cv2.resize(image, (512,512))

        if image is None:
            print(f"Failed to load {image_files[image_index]}")
            image_index += 1
            continue

        filename = image_files[image_index].split('.')[0]
        mask_filename = f"{filename}.png"
        
        canvas = image.copy()
        mask = np.zeros_like(image[:, :, 0])  # 建立與影像同尺寸的黑色畫布

        cv2.namedWindow('Drawing Canvas')
        cv2.setMouseCallback('Drawing Canvas', mouse_event)
        cv2.imshow('Drawing Canvas', canvas)

        while True:
            key = cv2.waitKey(5) & 0xFF

            if key == ord('q'):  # 退出
                cv2.destroyAllWindows()
                return
            elif key == ord('b'):  # 復原
                if undo_stack:
                    canvas, mask = undo_stack.pop()
                    cv2.imshow('Drawing Canvas', canvas)
                    print("Undo one step.")
                else:
                    print("Nothing to undo.")
            elif key == ord('1'):  # 細筆刷
                brush_index = 0
                print("Brush size: Small")
            elif key == ord('2'):  # 中筆刷
                brush_index = 1
                print("Brush size: Medium")
            elif key == ord('3'):  # 粗筆刷
                brush_index = 2
                print("Brush size: Large")
            elif key == ord('s'):  # 儲存 mask 並切換到下一張
                save_path = os.path.join(mask_dir, mask_filename)
                cv2.imwrite(save_path, mask)
                print(f"Saved mask: {save_path}")
                image_index += 1
                break
            elif key == ord('n'):  # 切換到下一張
                if image_index > 0:
                    image_index += 1
                    print(f"Next image: {image_files[image_index]}")
                    break
            elif key == ord('p'):  # 回到上一張
                if image_index > 0:
                    image_index -= 1
                    print(f"Back to previous image: {image_files[image_index]}")
                    break
                else:
                    print("Already at the first image.")

    cv2.destroyAllWindows()

def draw_np_v2(image_dir, mask_dir, image_index=0):
    os.makedirs(mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the source directory.")
        return

    brush_sizes = [20, 30, 50]
    brush_index = 0
    drawing = False
    points = []
    undo_stack = []
    drawing_mode = 'free'  # 'free' for freehand, 'line' for click-two-points
    line_points = []       # store two click points


    def mouse_event(event, x, y, flags, param):
        nonlocal drawing, points, canvas, mask, undo_stack, line_points

        if drawing_mode == 'free':
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                points = [[x, y]]
                undo_stack.append((canvas.copy(), mask.copy()))

            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                points.append([x, y])
                if len(points) >= 2:
                    cv2.line(canvas, tuple(points[-2]), tuple(points[-1]), (255, 255, 255, 255), brush_sizes[brush_index])
                    cv2.line(mask, tuple(points[-2]), tuple(points[-1]), (255), brush_sizes[brush_index])
                cv2.imshow('Drawing Canvas', canvas)

            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                if len(points) > 3:
                    # Check if path is closed: distance between start and end
                    dist = np.linalg.norm(np.array(points[0]) - np.array(points[-1]))
                    if dist < brush_sizes[brush_index] * 2:
                        print("Closed path detected. Filling shape...")
                        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], 255)
                        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
                points = []
                cv2.imshow('Drawing Canvas', canvas)

        elif drawing_mode == 'line':
            # --- Point-to-point line drawing ---
            if event == cv2.EVENT_LBUTTONDOWN:
                line_points.append((x, y))
                if len(line_points) == 2:
                    undo_stack.append((canvas.copy(), mask.copy()))
                    pt1, pt2 = line_points
                    cv2.line(canvas, pt1, pt2, (255, 255, 255, 255), brush_sizes[brush_index])
                    cv2.line(mask, pt1, pt2, (255, 255, 255), brush_sizes[brush_index])
                    cv2.imshow('Drawing Canvas', canvas)
                    line_points = []  # reset for next line


    while image_index < len(image_files):
        # Load image
        image_path = os.path.join(image_dir, image_files[image_index])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Failed to load {image_files[image_index]}")
            image_index += 1
            continue

        canvas = image.copy()
        mask = np.zeros_like(image[:, :, 0])

        filename = image_files[image_index].split('.')[0]
        mask_filename = f"{filename}.png"

        mask_path = os.path.join(mask_dir, mask_filename)

        # If a previous mask exists, load and draw it
        if os.path.exists(mask_path):
            print(f"Existing mask found: {mask_path}")
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, thresholded = cv2.threshold(loaded_mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours and draw them onto canvas and mask
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            cv2.drawContours(canvas, contours, -1, (255, 255, 255, 255), thickness=cv2.FILLED)

        # Setup window and mouse callback
        cv2.namedWindow('Drawing Canvas')
        cv2.setMouseCallback('Drawing Canvas', mouse_event)
        cv2.imshow('Drawing Canvas', canvas)

        while True:
            key = cv2.waitKey(5) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            
            elif key == ord('m'):
                if drawing_mode == 'free':
                    drawing_mode = 'line'
                    print("Switched to Line mode (click two points to draw line).")
                else:
                    drawing_mode = 'free'
                    print("Switched to Freehand mode (hold mouse and drag).")

            elif key in [ord('1'), ord('2'), ord('3')]:
                brush_index = int(chr(key)) - 1
                print(f"Brush size: {['Small', 'Medium', 'Large'][brush_index]}")
            
            elif key == ord('b'):
                if undo_stack:
                    canvas, mask = undo_stack.pop()
                    cv2.imshow('Drawing Canvas', canvas)
                    print("Undo one step.")
                else:
                    print("Nothing to undo.")

            elif key == ord('s'):
                save_path = os.path.join(mask_dir, mask_filename)
                cv2.imwrite(save_path, mask)
                print(f"Saved mask: {save_path}")
                image_index += 1
                break

            elif key == ord('n'):
                image_index += 1
                if image_index < len(image_files):
                    print(f"Next image: {image_files[image_index]}")
                break

            elif key == ord('p'):
                if image_index > 0:
                    image_index -= 1
                    print(f"Back to previous image: {image_files[image_index]}")
                else:
                    print("Already at the first image.")
                break

            elif key == ord('d'):
                # Discard existing mask and reset to clean canvas
                canvas = image.copy()
                mask = np.zeros_like(image[:, :, 0])
                undo_stack.clear()  # Also clear undo history to avoid conflicts
                print("Discarded previous mask. Starting new drawing.")
                cv2.imshow('Drawing Canvas', canvas)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive tool to annotate defect masks on PCB images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Keyboard shortcuts:
  1/2/3     - Set brush size (small/medium/large)
  S         - Save current mask and move to next image
  N/P       - Next/previous image
  B         - Undo last stroke
  M         - Toggle between freehand and line drawing mode
  D         - Discard current mask and start fresh
  Q         - Quit tool

Examples:
  # Annotate all classes
  python pre_mask_utils.py --data_dir ./data

  # Annotate specific class
  python pre_mask_utils.py --data_dir ./data --class_name scratch

  # Start from image 5
  python pre_mask_utils.py --data_dir ./data --start_image 5
        '''
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to data directory (must contain images/ and masks/ folders)'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default=None,
        help='Specific class to annotate (e.g., "scratch"). If None, annotates all classes.'
    )
    parser.add_argument(
        '--start_image',
        type=int,
        default=0,
        help='Image index to start from (default: 0)'
    )
    parser.add_argument(
        '--version',
        type=str,
        choices=['v1', 'v2'],
        default='v2',
        help='Version of drawing tool (v1: basic, v2: advanced with line mode and shape filling)'
    )

    args = parser.parse_args()

    image_dir = os.path.join(args.data_dir, "images")
    mask_dir = os.path.join(args.data_dir, "masks")

    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} does not exist!")
        sys.exit(1)

    # Choose drawing function
    draw_func = draw_np_v2 if args.version == 'v2' else draw_np

    if args.class_name:
        # Annotate specific class
        class_image_dir = os.path.join(image_dir, args.class_name)
        class_mask_dir = os.path.join(mask_dir, args.class_name)

        if not os.path.exists(class_image_dir):
            print(f"Error: {class_image_dir} does not exist!")
            sys.exit(1)

        print(f"Annotating class: {args.class_name}")
        draw_func(image_dir=class_image_dir, mask_dir=class_mask_dir, image_index=args.start_image)
    else:
        # Annotate all classes
        classes = sorted(os.listdir(image_dir))
        print(f"Found {len(classes)} classes: {classes}")

        for class_name in classes:
            class_image_dir = os.path.join(image_dir, class_name)
            class_mask_dir = os.path.join(mask_dir, class_name)

            if os.path.isdir(class_image_dir):
                print(f"\n>>> Annotating class: {class_name}")
                draw_func(image_dir=class_image_dir, mask_dir=class_mask_dir, image_index=args.start_image)

    print("Mask annotation complete!")