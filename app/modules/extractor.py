import cv2
import dlib
import mediapipe as mp
from ..utils import utils
from ..settings import config


def extract_face_features_dlib(image_path, model_name, image_basename, keypoint_color, margin_ratio):
    model_predictor = config.APP_PATH_MODEL_DLIB_FILE
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
        x_min, x_max, y_min, y_max = utils.extract_bounding_box_dlib(landmarks, 68, image.shape[1], image.shape[0], margin_ratio)
        x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

        # 1:1
        width = x_max - x_min
        height = y_max - y_min
        max_side = max(width, height)
        x_min = max(x_min - (max_side - width) // 2, 0)
        y_min = max(y_min - (max_side - height) // 2, 0)
        x_max = min(x_min + max_side, image.shape[1])
        y_max = min(y_min + max_side, image.shape[0])

        face_image_with_points = image[y_min:y_max, x_min:x_max].copy()
        face_image_without_points = face_image_with_points.copy()

        for n in range(0, 68):
            x = landmarks.part(n).x - x_min
            y = landmarks.part(n).y - y_min

            if 0 <= x < face_image_with_points.shape[1] and 0 <= y < face_image_with_points.shape[0]:
                cv2.circle(face_image_with_points, (x, y), 1, keypoint_color, -1)

        output_dir = utils.create_output_directory(model_name)

        # utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_with_points.jpg', face_image_with_points)
        utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_without_points.jpg', face_image_without_points)
        
        face_count += 1

def extract_face_features_mediapipe(image_path, model_name, image_basename, keypoint_color, margin_ratio):
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
            x_min, x_max, y_min, y_max = utils.extract_bounding_box_mediapipe(face_landmarks, image.shape, margin_ratio)
            x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

            # 1:1
            width = x_max - x_min
            height = y_max - y_min
            max_side = max(width, height)
            x_min = max(x_min - (max_side - width) // 2, 0)
            y_min = max(y_min - (max_side - height) // 2, 0)
            x_max = min(x_min + max_side, image.shape[1])
            y_max = min(y_min + max_side, image.shape[0])

            face_image_with_points = image[y_min:y_max, x_min:x_max].copy()
            face_image_without_points = face_image_with_points.copy()

            for lm in face_landmarks.landmark:
                x = int(lm.x * image.shape[1]) - x_min
                y = int(lm.y * image.shape[0]) - y_min

                if 0 <= x < face_image_with_points.shape[1] and 0 <= y < face_image_with_points.shape[0]:
                    cv2.circle(face_image_with_points, (x, y), 1, keypoint_color, -1)

            output_dir = utils.create_output_directory(model_name)

            # utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_with_points.jpg', face_image_with_points)
            utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_without_points.jpg', face_image_without_points)

            face_count += 1

    face_mesh.close()
