import numpy as np
import ollama
import cv2
import os
import pandas as pd



class Config:
    # Paths (Update accordingly)
    train_base_dir = ""
    train_csv_path = ""

    val_base_dir = ""
    val_csv_path = ""

    # Model configuration
    modalities = {"camera": True, "gps": True, "lidar": True}

    # Training configuration
    batch_size = 12
    learning_rate = 0.0001
    # num_epochs = 10
    num_epochs = 100


def extract_scenario(file_path):
    # Assuming scenario is in the format "./scenarioXX/" in the file paths
    parts = file_path.split('/')
    for part in parts:
        if part.startswith("scenario"):
            return part
    return "unknown"  # Default if no scenario is found


def get_index(file_path):
    # Extract the base index from the existing file path
    parts = file_path.split('_')
    try:
        base_index = int(parts[-1].split('.')[0])  # Extract number before ".txt"
        return base_index
    except (IndexError, ValueError):
        return None  # Return None if parsing fails

class CameraDataset():
    def __init__(self, csv_path, base_dir):
        """
        Initializes the dataset with paths from CSV and a base directory.

        Parameters:
            csv_path (str): Path to the CSV file containing relative file paths.
            base_dir (str): Base directory where data files are located (e.g., Multi_Modal or Multi_Modal_test).
        """
        self.data = pd.read_csv(csv_path)
        self.base_dir = base_dir

    def load_image(self, img_path):
        full_img_path = os.path.join(self.base_dir, img_path)  # Construct full path
        if not os.path.isfile(full_img_path):
            raise FileNotFoundError(f"Image file not found: {full_img_path}")

        image = cv2.imread(full_img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {full_img_path}")

        # image = cv2.resize(image, (256, 256))
        return image

    def get_paths(self, idx):
        row = self.data.iloc[idx]
        camera_paths = [row[f"unit1_rgb_{i}"] for i in range(1, 6)]
        return camera_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        camera_data = [self.load_image(row[f"unit1_rgb_{i}"]) for i in range(1, 6)]
        return camera_data


def image_to_text(image_file):
    prompt = "Analyze the given image and describe its contents as accurately and literally as possible. Avoid any metaphorical or interpretive language. Focus solely on describing what is visually present in the scene, including objects, people, actions, settings, and any noticeable details. Be straightforward and detailed in your description."
    response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_file]
                }]
            )

    return response['message']['content']


def main():
    config = Config()

    csv_path = os.path.join('./', config.train_csv_path)
    # print(csv_path)
    base_dir_path = os.path.join('./', config.train_base_dir)
    dataset = CameraDataset(csv_path=csv_path, base_dir=base_dir_path)

    paths = dataset.get_paths(0)[0].split('./', 1)[1]
    # update accordingly
    img_path = os.path.join(base_dir_path, paths)

    img = cv2.imread((paths))
    description = image_to_text(img_path)
    print(description)


if __name__ == '__main__':
    main()
