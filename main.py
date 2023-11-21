import os
import glob
import cv2
import dlib
import mediapipe as mp

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

def extract_face_features_dlib(image_path, model_name, image_basename, keypoint_color=(255, 0, 0)):
    model_predictor = 'models/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_predictor)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print(f"No faces found in the image for model {model_name}.")
        return

    face_count = 0

    for face in faces:
        landmarks = predictor(gray, face)
        x_min, x_max, y_min, y_max = extract_bounding_box_dlib(landmarks, 68, image.shape[1], image.shape[0])
        x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

        face_image_with_points = image[y_min:y_max, x_min:x_max].copy()
        face_image_without_points = face_image_with_points.copy()

        for n in range(0, 68):
            x = landmarks.part(n).x - x_min
            y = landmarks.part(n).y - y_min

            if 0 <= x < face_image_with_points.shape[1] and 0 <= y < face_image_with_points.shape[0]:
                cv2.circle(face_image_with_points, (x, y), 1, keypoint_color, -1)

        output_dir = create_output_directory(model_name)
        cv2.imwrite(f'{output_dir}/{image_basename}_face_{face_count}_with_points.jpg', face_image_with_points)
        cv2.imwrite(f'{output_dir}/{image_basename}_face_{face_count}_without_points.jpg', face_image_without_points)
        face_count += 1

def extract_face_features_mediapipe(image_path, model_name, image_basename, keypoint_color=(255, 0, 0)):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_count = 0

        for face_landmarks in results.multi_face_landmarks:
            x_min, x_max, y_min, y_max = extract_bounding_box_mediapipe(face_landmarks, image.shape)
            x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

            face_image_with_points = image[y_min:y_max, x_min:x_max].copy()
            face_image_without_points = face_image_with_points.copy()

            for lm in face_landmarks.landmark:
                x = int(lm.x * image.shape[1]) - x_min
                y = int(lm.y * image.shape[0]) - y_min

                if 0 <= x < face_image_with_points.shape[1] and 0 <= y < face_image_with_points.shape[0]:
                    cv2.circle(face_image_with_points, (x, y), 1, keypoint_color, -1)

            output_dir = create_output_directory(model_name)
            cv2.imwrite(f'{output_dir}/{image_basename}_face_{face_count}_with_points.jpg', face_image_with_points)
            cv2.imwrite(f'{output_dir}/{image_basename}_face_{face_count}_without_points.jpg', face_image_without_points)
            face_count += 1

    face_mesh.close()

if __name__ == '__main__':
    image_directory = 'images'
    models = ['dlib', 'mediapipe']
    images_format = ['.png', '.jpg', '.jpeg']

    keypoint_color = (255, 255, 255)

    image_paths = []
    for img_format in images_format:
        image_paths.extend(glob.glob(os.path.join(image_directory, '*' + img_format)))

    for image_path in image_paths:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            for model in models:
                if model == 'dlib':
                    extract_face_features_dlib(image_path, model, image_basename, keypoint_color)
                elif model == 'mediapipe':
                    extract_face_features_mediapipe(image_path, model, image_basename, keypoint_color)
        except Exception as e:
            print(f'Error with image: {image_path}')
            print(e)
