import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

hc = []

# Function to extract video frames
def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not os.path.exists(majorData):
        os.makedirs(majorData)

    # If gesture_folder is a file, process it directly
    if os.path.isfile(gesture_folder):
        videos = [gesture_folder]
    else:
        gesture_folder = os.path.abspath(gesture_folder)
        videos = [os.path.join(gesture_folder, video) for video in os.listdir(gesture_folder) if os.path.isfile(os.path.join(gesture_folder, video))]

    for video in tqdm(videos, unit='videos', ascii=True):
        video_name = os.path.abspath(video)
        cap = cv2.VideoCapture(video_name)
        gesture_frames_path = os.path.join(majorData, os.path.splitext(os.path.basename(video))[0])
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            framename = f"{gesture_frames_path}/frame_{count}.jpeg"
            hc.append([framename, 'gesture'])
            cv2.imwrite(framename, frame)
            count += 1
        cap.release()
    os.chdir(rootPath)

# Function to resize frames
def resize_frames(input_path, output_path, size=224):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in glob.glob(input_path + '**/*.jpeg', recursive=True):
        img = Image.open(filename).resize((size, size))
        loc = os.path.split(filename)[0]
        subdir = loc.split('/')[-1]  # Adjust for '/' or '\\' as needed
        fullnew_subdir = os.path.join(output_path, subdir)
        if not os.path.exists(fullnew_subdir):
            os.makedirs(fullnew_subdir)
        name = os.path.split(filename)[1]
        img.save(os.path.join(fullnew_subdir, name))

# Function to extract holistic landmarks
def extract_holistic_landmarks(frame, holistic):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    landmarks = {
        'hand_landmarks': np.array([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None,
        'face_landmarks': np.array([(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark]) if results.face_landmarks else None,
        'pose_landmarks': np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else None,
    }
    return landmarks

# Main integration
if __name__ == "__main__":
    gesture_folder = '/path/to/gesture_folder'
    target_folder = '/path/to/target_folder'
    resized_folder = '/path/to/resized_frames'

    # Step 1: Extract frames
    convert(gesture_folder, target_folder)

    # Step 2: Resize extracted frames
    resize_frames(target_folder, resized_folder)

    # Step 3: Extract landmarks
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)
    for img_path in glob.glob(resized_folder + '**/*.jpeg', recursive=True):
        frame = cv2.imread(img_path)
        landmarks = extract_holistic_landmarks(frame, holistic)
        print(f"Processed {img_path}: {landmarks}")
