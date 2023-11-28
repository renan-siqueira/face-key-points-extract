import math
import cv2
import dlib
import mediapipe as mp
import numpy as np
from ..utils import utils
from ..settings import config


def align_face_dlib(image, landmarks):
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image


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

        image_aligned = align_face_dlib(image, landmarks)

        gray_aligned = cv2.cvtColor(image_aligned, cv2.COLOR_BGR2GRAY)
        
        faces_aligned = detector(gray_aligned)

        if len(faces_aligned) > 0:
            face_aligned = faces_aligned[0]
            landmarks_aligned = predictor(gray_aligned, face_aligned)
        
            x_min, x_max, y_min, y_max = utils.extract_bounding_box_dlib(landmarks_aligned, 68, image_aligned.shape[1], image_aligned.shape[0], margin_ratio)
            x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

            # 1:1
            width = x_max - x_min
            height = y_max - y_min
            max_side = max(width, height)

            x_min = max(x_min - (max_side - width) // 2, 0)
            y_min = max(y_min - (max_side - height) // 2, 0)
            x_max = min(x_min + max_side, image_aligned.shape[1])
            y_max = min(y_min + max_side, image_aligned.shape[0])

            face_image_with_points = image_aligned[y_min:y_max, x_min:x_max].copy()
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
        else:
            print(f"No aligned faces found in the image for model {model_name}.")


def _get_landmark_point(face_landmarks, index, image_shape):
    x = int(face_landmarks.landmark[index].x * image_shape[1])
    y = int(face_landmarks.landmark[index].y * image_shape[0])
    return (x, y)


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

            # Alinhamento da face
            eye_left = _get_landmark_point(face_landmarks, 130, image.shape) # ponto do olho esquerdo
            eye_right = _get_landmark_point(face_landmarks, 359, image.shape) # ponto do olho direito
            angle = math.atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
            degrees = math.degrees(angle)
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            M = cv2.getRotationMatrix2D(center, degrees, 1)
            aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            face_image_with_points = aligned_image[y_min:y_max, x_min:x_max].copy()
            face_image_without_points = face_image_with_points.copy()

            for lm in face_landmarks.landmark:
                x = int(lm.x * aligned_image.shape[1]) - x_min
                y = int(lm.y * aligned_image.shape[0]) - y_min

                if 0 <= x < face_image_with_points.shape[1] and 0 <= y < face_image_with_points.shape[0]:
                    cv2.circle(face_image_with_points, (x, y), 1, keypoint_color, -1)

            output_dir = utils.create_output_directory(model_name)
            # utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_with_points.jpg', face_image_with_points)
            utils.save_image(f'{output_dir}/{image_basename}_face_{face_count}_without_points.jpg', face_image_without_points)

            face_count += 1

    face_mesh.close()
