import os
import glob
from .modules import extractor


def execute(params):
    image_paths = []
    for img_format in params.get("images_format", []):
        image_paths.extend(glob.glob(os.path.join(params.get("input_path", None), '*' + img_format)))

    for image_path in image_paths:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            for model in params.get("models", []):
                if model == 'dlib':
                    extractor.extract_face_features_dlib(image_path, model, image_basename, params.get("keypoints_color", None))
                elif model == 'mediapipe':
                    extractor.extract_face_features_mediapipe(image_path, model, image_basename, params.get("keypoints_color", None))
        except Exception as e:
            print(f'Error with image: {image_path}')
            print(e)
