import os
from PIL import Image


def convert_to_square(image_path, output_path):
    """ Convert an image to a 1:1 aspect ratio by cropping it from the center. """
    with Image.open(image_path) as img:
        width, height = img.size

        new_size = min(width, height)

        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(output_path)

def process_folder(input_folder, output_folder):
    """ Process all images in a folder, converting them to 1:1 aspect ratio. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_to_square(file_path, output_path)


def main():
    input_folder = 'images'
    output_folder = 'square'

    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
