import os
import shutil


def organize_images(base_path):
    for root, _, files in os.walk(base_path):

        points_path = os.path.join(root, 'points')
        original_path = os.path.join(root, 'original')

        if not os.path.exists(points_path):
            os.makedirs(points_path)
        if not os.path.exists(original_path):
            os.makedirs(original_path)

        for file in files:
            if '_with_points.' in file:
                shutil.move(os.path.join(root, file), points_path)
            elif '_without_points.' in file:
                shutil.move(os.path.join(root, file), original_path)


if __name__ == '__main__':
    base_path = 'output'
    organize_images(base_path)
