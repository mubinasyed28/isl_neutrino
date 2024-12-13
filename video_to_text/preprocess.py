import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import os
import numpy as np

frame_folder = "/Users/shriya/Documents/GitHub/isl_neutrino/target_folder/newvideo"
frame_files = [f for f in os.listdir(frame_folder) if os.path.isfile(os.path.join(frame_folder, f))]

for frame_file in frame_files:
    frame_path = os.path.join(frame_folder, frame_file)

    img = cv2.imread(frame_path)

    if img is None:
        print(f"Unable to read the frame: {frame_file}")
        continue

    bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

    # Wait for a key press and close windows
    key = cv2.waitKey(0)
    if key == 27:  # Press 'Esc' to exit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
