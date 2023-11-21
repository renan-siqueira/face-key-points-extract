import os
import cv2


def create_output_directory(model_name):
    directory = f'output/{model_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def add_margin_to_bounding_box(x_min, x_max, y_min, y_max, width, height, margin_ratio=0.5):
    x_margin = int((x_max - x_min) * margin_ratio)
    y_margin = int((y_max - y_min) * margin_ratio)

    x_min = max(0, x_min - x_margin)
    y_min = max(0, y_min - y_margin)
    x_max = min(width, x_max + x_margin)
    y_max = min(height, y_max + y_margin)

    return x_min, x_max, y_min, y_max

def extract_bounding_box_dlib(landmarks, num_points, image_width, image_height):
    x_min = min([landmarks.part(n).x for n in range(num_points)])
    x_max = max([landmarks.part(n).x for n in range(num_points)])
    y_min = min([landmarks.part(n).y for n in range(num_points)])
    y_max = max([landmarks.part(n).y for n in range(num_points)])

    return add_margin_to_bounding_box(x_min, x_max, y_min, y_max, image_width, image_height)

def extract_bounding_box_mediapipe(face_landmarks, image_shape):
    width, height = image_shape[1], image_shape[0]
    x_min = min([lm.x for lm in face_landmarks.landmark]) * width
    y_min = min([lm.y for lm in face_landmarks.landmark]) * height
    x_max = max([lm.x for lm in face_landmarks.landmark]) * width
    y_max = max([lm.y for lm in face_landmarks.landmark]) * height

    return add_margin_to_bounding_box(x_min, x_max, y_min, y_max, width, height)

def save_image(path, image):
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(f'Error when saving image: {path}')
        print(e)
