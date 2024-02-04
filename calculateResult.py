import os
import cv2
import pandas as pd
import numpy as np
import re
from openpyxl import Workbook
import math

def get_id_from_filename(filename):
    match = re.search(r"RET(\d+)(OD|OS)", filename)
    if match:
        return (match.group(1).zfill(3), match.group(2))
    return None

def calculate_score(image, axial_length):

    if math.isnan(axial_length):
        return 0
    
    ratio = axial_length / 26
    crop_size = int(128 * ratio)
    x = (image.shape[1] - crop_size) // 2
    y = (image.shape[0] - crop_size) // 2
    cropped_image = image[y:y+crop_size, x:x+crop_size]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels = np.sum(binary_image == 255)
    total_pixels = binary_image.size
    score = white_pixels / total_pixels

    return score

def process_dataframe(dataframe, input_folder, level, eye_type):
    dataframe["image"] = ""
    dataframe["score"] = 0.0

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_name}")
            continue

        img_info = get_id_from_filename(img_name)

        if img_info is None:
            print(f"Failed to extract ID from image name: {img_name}")
            continue

        img_id, img_type = img_info
        print(img_id, img_type, eye_type, img_type==eye_type)

        if img_type == eye_type:
            dataframe[(level,'ID')] = dataframe[(level,'ID')].str.replace('#', '').astype(int)
            print("----------------------")
            print(dataframe.head(5))
            print(dataframe[(level,'ID')])
            row_index = dataframe.index[dataframe[((level, "ID"))] == img_id].tolist()

            if len(row_index) == 1:
                dataframe.at[row_index[0], "image"] = img_name            
                axial_length = dataframe.loc[row_index[0], ('Axial_Length', 'Axial_Length')]
                dataframe.at[row_index[0], "score"] = calculate_score(img, axial_length)
            else:
                print(f"No matching row found for image: {img_name}")

    return dataframe

od_data = pd.read_excel('od.xlsx', header=[0, 1])
os_data = pd.read_excel('Datasets for Data Cleaning and Analysis/os.xlsx', header=[0, 1])


input_folder = "image-processing-files/test_images"

od_data = process_dataframe(od_data, input_folder,"Unnamed: 1_level_0","OD")
os_data = process_dataframe(os_data, input_folder,"Unnamed: 0_level_0","OS")

od_data.to_excel('od_score.xlsx', index=True, engine='openpyxl')
os_data.to_excel('os_score.xlsx', index=True, engine='openpyxl')

print(od_data.head(5))

od_data_is = od_data.loc[:, [('image', ''), ('score', '')]]
od_data_is = od_data_is[od_data_is[('image', '')] != '']
os_data_is = os_data.loc[:, [('image', ''), ('score', '')]]
os_data_is = os_data_is[os_data_is[('image', '')] != '']
# Concatenating the two dataframes vertically (i.e. stacking them on top of each other)
merged_df = pd.concat([od_data_is, os_data_is])

# Resetting the index of the merged dataframe
merged_df = merged_df.reset_index(drop=True)

merged_df.to_csv('score_results.csv', index=False)
