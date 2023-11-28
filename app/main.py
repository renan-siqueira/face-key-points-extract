import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .modules import extractor


def process_image(image_path, params):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    keypoints_color = params.get("keypoints_color", (0, 0, 255))
    margin_ratio = params.get("margin_ratio", 0.5)
    try:
        for model in params.get("models", []):
            if model == 'dlib':
                extractor.extract_face_features_dlib(
                    image_path, model, image_basename, keypoints_color, margin_ratio
                )
            elif model == 'mediapipe':
                extractor.extract_face_features_mediapipe(
                    image_path, model, image_basename, keypoints_color, margin_ratio
                )
    except Exception as e:
        print(f'Error with image: {image_path}')
        print(e)


def execute(params, debug_mode=False):
    num_threads = params.get('num_threads', 4)
    print('\n---> Threads used:', num_threads, '\n')

    image_paths = []
    input_path = 'images' if debug_mode else params['input_path']
    for img_format in params.get("images_format", []):
        image_paths.extend(
            glob.glob(os.path.join(input_path, '*' + img_format))
        )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for image_path in image_paths:
            future = executor.submit(process_image, image_path, params)
            futures[future] = image_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            try:
                future.result()
            except Exception as e:
                image_path = futures[future]
                print(f'Error with image: {image_path}')
                print(e)
