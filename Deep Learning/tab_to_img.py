"""
This file contains methods used for converting tabular data to images.

A Conversion Method Should:
    1. Return a 4 dimensional ndarray (n_samples, height, row, channel).
    2. Return scaled values
    3. Return with the smallest possible dtype
       -> For grayscale images: np.float16
       -> For black/white: np.uint8

"""
# Common Imports
import pandas as pd
import cv2 as cv
import struct
import numpy as np
import os
from scipy.stats import pearsonr

from sklearn.preprocessing import MinMaxScaler

#########################################
##                                     ##
##   Binary Image Encoding             ##
##                                     ##
#########################################

def bie(
        data: pd.DataFrame,
        repeat: int = 1,
        image_width: int = 64,
        cache: tuple[bool, str] = (True, "bie.npy"),
        override: bool = False,
        show_example_image: bool = False
):

    (cache, file) = cache
    if not override:
        if cache:
            if os.path.exists(file):
                X = np.load(file)
                y = data.iloc[:, -1].to_numpy()
                return X, y


    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    images = []
    for row in X:
        bin_arr = []
        for num in row:
            bin_val = [int(bit) for bit in ''.join(f'{byte:08b}' for byte in struct.pack('d', num))]
            for _ in range(repeat):
                bin_arr.append(bin_val)
        bin_img = np.array(bin_arr, dtype=np.uint8)[:, :image_width]
        images.append(bin_img.reshape(-1, image_width, 1))
    
    images = np.array(images, dtype=np.uint8)

    if show_example_image:
        cv.imshow("Binary Encoded Image", images[0] * 255)
        cv.waitKey(0)

    if cache:
        np.save(file, images)

    return images, y


#########################################
##                                     ##
##   Dynamic Weighted Tabular Method   ##
##                                     ##
#########################################

def dwtm(
        data: pd.DataFrame, 
        max_str_length: int = 50,
        max_font_scale: float = 1,
        min_font_scale: float = 0.3,
        color: tuple[int, int, int] = (255, 255, 255),
        font_face: int = cv.FONT_HERSHEY_SIMPLEX,
        thickness: int = 1,
        line_type:int = cv.LINE_4,
        cache: tuple[bool, str] = (True, "dwtm.npy"),
        override: bool = False,
        show_example_image: bool = False
):
    """
    This method performs a modified version of DWTM for converting tabular data to images.

    The paper that introduced the DWTM algorithm can be found here:
    https://arxiv.org/abs/2205.10386

    MLA Citation:
    Iqbal, Md Ifraham, et al. "A Dynamic Weighted Tabular Method for Convolutional Neural Networks." arXiv preprint arXiv:2205.10386 (2022).

    The original code that this file is adapted from can be found here:
    https://github.com/Ifraham/Dynamic-Weighted-Tabular-Method

    """
    (cache, file) = cache
    if not override:
        if cache:
            if os.path.exists(file):
                X = np.load(file)
                y = data.iloc[:, -1].to_numpy()
                return X, y

    # Split Dataset
    X = data.iloc[:, :-1].to_numpy()
    labels = data.iloc[:, -1].to_numpy()

    # Get Number of Features
    tot_features = len(X[0])

    # Calculate R Scores
    r_scores = []
    for i in range(tot_features):
        r, _ = pearsonr(X[:, i], labels)
        r_scores.append(abs(r))
    r_scores = np.array(r_scores)

    # Sort Based on R Scores
    sorted_index = np.argsort(r_scores)[::-1]
    r_scores = r_scores[sorted_index]
    X = X[:, sorted_index]

    # Convert R Scores to Font Scales
    scaler = MinMaxScaler((min_font_scale, max_font_scale))
    font_scales = scaler.fit_transform(r_scores.reshape(-1, 1)).reshape(-1)
    
    # Find Max String Lenghts of each column
    char_lens = []
    for i in range(tot_features):
        max_length = np.vectorize(lambda x: len(str(x)))(X[:, i]).max()
        char_lens.append(min(max_length, max_str_length))
    char_lens = np.array(char_lens)

    # Find pixels sizes for each feature
    boxes = []
    for fs, l in zip(font_scales, char_lens):
        s = "".join(["1" for _ in range(l)])
        res = cv.getTextSize(s, font_face, fs, thickness)
        print(res)
        res = ((res[0][0], res[0][1]), res[1])
        boxes.append(res)

    width = max(map(lambda box: box[0][0], boxes))
    height = 0

    points = []
    available_points = []
    
    # Only Origin Available Initially
    available_points.append((0, 0))

    # Insert Each Box
    for (w, h), _ in boxes:
        idx = -1
        x = -1
        y = -1
        for i, (px, py) in enumerate(available_points):
            idx = i
            if w + px <= width:
                points.append((px, py + h))
                x = px
                y = py
                break
        
        height = max(height, y + h)
        available_points.pop(idx)
        available_points.insert(idx, (x + w, y))
        available_points.insert(idx + 1, (x, y + h))
        available_points = sorted(available_points, key=lambda box: box[1]) # sort by height

    images = []
    for sample in X:
        img = np.zeros((height, width), dtype=np.uint8)
        for val, rs, (x, y), cl in zip(sample, font_scales, points, char_lens):
            cv.putText(img, str(val)[0:cl], (x, y), font_face, rs, color, thickness, line_type)
        images.append(img)
    images = np.array(images, dtype=np.uint8).reshape((-1, height, width, 1)) / 255
    images = images.astype(np.uint8)

    if show_example_image:
        cv.imshow("Binary Encoded Image", images[0] * 255)
        cv.waitKey(0)

    if cache:
        np.save(file, images)

    return images, labels



