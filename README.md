# Face Key Points Extractor

## Overview

This project is a Python application designed for image processing with a focus on face feature extraction. 
It uses libraries like `dlib` and `mediapipe` for advanced face detection and processing.

---

## Installation

1. Clone the repository.

2. Navigate to the project directory.

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

- On Windows:

```bash
.venv\Scripts\activate
```

- On Unix or MacOS:

```bash
source .venv/bin/activate
```

5. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

### Dlib Model Configuration

Before running the application, download the `shape_predictor_68_face_landmarks.dat` file for dlib:

1. Download the model file from [Dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) or a trusted source.

2. Unzip the file and place it in the `models` directory at the root of the project.

3. Ensure the `config.py` in the `settings` directory is set correctly:

```python
APP_PATH_PARAMS_JSON_FILE = 'params.json'
APP_PATH_MODEL_DLIB_FILE = 'models/shape_predictor_68_face_landmarks.dat'
```

---

### Configuration parameters file

Customize the application behavior by modifying the `params.json` file in the root directory:

```json
{
    "input_path": "",
    "models": [
        "dlib",
        "mediapipe"
    ],
    "images_format": [
        ".png",
        ".jpg",
        ".jpeg"
    ],
    "keypoints_color": [255, 255, 255],
    "margin_ratio": 0.1,
    "num_threads": 8
}
```

- `input_path`: Path to the directory containing the images to be processed.

- `models`: List of models to use for face feature extraction. Supported models are **dlib** and **mediapipe**.

- `images_format`: List of image file formats to be processed. For example: `.png`, `.jpg`, `.jpeg`.

- `keypoints_color`: Color of the keypoints (in BGR format) to be drawn on the images. It's an array of three integers representing Blue, Green, and Red color values.

- `margin_ratio`: The margin ratio around the detected face for cropping. A smaller value results in a tighter crop around the face.

- `num_threads`: The number of threads to use for image processing. Increasing this number can speed up processing on multi-core systems.

---

## Usage

To run the application, use the following command:

```bash
python run.py [--debug]
```

The `--debug` argument is optional and enables the debug mode for additional logging and process details.

## Key Components

- `run.py`: The entry point of the application. It parses command-line arguments and initializes the application with the given parameters.

- `main.py`: Contains the main logic for processing images using multithreading.

- `extractor.py`: Includes functions for face feature extraction using `dlib` and `mediapipe`.

- `utils.py`: Provides utility functions used across the application.

- `config.py`: Contains configuration settings for the application.

---

## License

This project is open-sourced and available to everyone under the [MIT License](LICENSE).
