import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder

class VideoDataGenerator(Sequence):
    def __init__(self, directory, batch_size, frame_count, frame_height, frame_width, num_classes):
        self.directory = directory
        self.batch_size = batch_size
        self.frame_count = frame_count
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_classes = num_classes
        self.video_files = []
        self.labels = []

        # Walk through each class directory within the base directory
        for class_dir in os.listdir(directory):
            class_path = os.path.join(directory, class_dir)
            if os.path.isdir(class_path):
                for f in os.listdir(class_path):
                    if f.endswith('.mp4'):
                        self.video_files.append(os.path.join(class_path, f))
                        self.labels.append(class_dir)  # Use directory name as label

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return int(np.ceil(len(self.video_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self._get_data(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_labels = self.encoded_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        return np.array(batch_x), np.array(batch_y)
    
    def _get_data(self, start_idx, end_idx):
        batch_x = []
        for idx in range(start_idx, min(end_idx, len(self.video_files))):
            video_path = self.video_files[idx]
            video_frames = self.load_video(video_path)
            if video_frames is not None and len(video_frames.shape) == 4:  # Ensure frames are as expected
                batch_x.append(video_frames)
        return np.array(batch_x)

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():  # Check if the video is successfully opened
            print(f"Failed to open video: {video_path}")
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))  # Resize frame
            frames.append(frame)
        cap.release()
        frames = np.array(frames, dtype=np.uint8)  # Ensure dtype matches OpenCV's default
        # Inside the load_video method, right before returning 'frames':
        if frames.shape[0] < self.frame_count:
            # Add padding
            padding = np.zeros((self.frame_count - frames.shape[0], self.frame_height, self.frame_width, 3), dtype=np.uint8)
            frames = np.concatenate((frames, padding), axis=0)
        elif frames.shape[0] > self.frame_count:
            frames = frames[:self.frame_count]

        assert frames.shape == (self.frame_count, self.frame_height, self.frame_width, 3), "Incorrect video shape for video => " + video_path

        return frames


def build_model(num_classes, frame_count, frame_height, frame_width):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation="relu", input_shape=(frame_count, frame_height, frame_width, 3)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

# Parameters
base_dir = "/home/amsozzer/Projects/Projector/DATA_SET"  # Update this path to your data directory
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
batch_size = 16
frame_count = 16
frame_height = 160
frame_width = 160
num_classes = 2  # Update this based on your actual number of classes

# Instantiate the data generators
train_generator = VideoDataGenerator(train_dir, batch_size, frame_count, frame_height, frame_width, num_classes)
val_generator = VideoDataGenerator(val_dir, batch_size, frame_count, frame_height, frame_width, num_classes)

# Build and compile the model
model = build_model(num_classes, frame_count, frame_height, frame_width)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=8)
