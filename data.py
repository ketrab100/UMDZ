import os

import numpy as np
import pandas as pd
from PIL import Image

emotions_encodings = {
    "angry": np.array([0, 0, 0, 0, 0, 1]),
    "fear": np.array([0, 0, 0, 0, 1, 0]),
    "happy": np.array([0, 0, 0, 1, 0, 0]),
    "neutral": np.array([0, 0, 1, 0, 0, 0]),
    "sad": np.array([0, 1, 0, 0, 0, 0]),
    "surprise": np.array([1, 0, 0, 0, 0, 0]),
}

emotion_labels = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def load_emotion(folder_path, emotion):
    image_files = sorted([f for f in os.listdir(folder_path)])
    rows = []

    for image_file in image_files:
        file_path = os.path.join(folder_path, image_file)

        image = Image.open(file_path).resize((48, 48))
        pixels = np.array(image).reshape(48, 48, 1)

        flipped_pixels = np.array(image.transpose(Image.FLIP_LEFT_RIGHT)).reshape(
            48, 48, 1
        )

        rows.append(
            {
                "file_name": file_path,
                "pixels": pixels,
                "label": emotion,
                "encoding": emotions_encodings[emotion],
            }
        )

        if emotion != "happy":
            rows.append(
                {
                    "file_name": f"{file_path}_flipped",
                    "pixels": flipped_pixels,
                    "label": emotion,
                    "encoding": emotions_encodings[emotion],
                }
            )

    df = pd.DataFrame(rows)
    return df


class Data:
    test_data = pd.DataFrame(columns=["file_name", "pixels", "label", "encoding"])
    train_data = pd.DataFrame(columns=["file_name", "pixels", "label", "encoding"])

    def __init__(self):
        self.load_train_data()
        self.load_test_data()

    def load_test_data(self):
        for i, emotion in enumerate(emotion_labels):
            folder_path = "archive/test/" + emotion + "/"
            self.test_data = pd.concat(
                [self.test_data, load_emotion(folder_path, emotion)]
            )
        return self.test_data

    def load_train_data(self):
        for i, emotion in enumerate(emotion_labels):
            folder_path = "archive/train/" + emotion + "/"
            self.train_data = pd.concat(
                [self.train_data, load_emotion(folder_path, emotion)]
            )
        return self.train_data
