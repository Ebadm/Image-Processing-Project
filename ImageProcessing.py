import cv2
import numpy as np
from skimage import restoration
from skimage.transform import warp, ProjectiveTransform
import os
import argparse

def denoise_image(image):
    # Apply Non-local Means denoising for Gaussian noise
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Apply Median filter for salt/pepper noise
    denoised_image = cv2.medianBlur(denoised_image, 5)
    return denoised_image

def enhance_contrast_and_brightness(image):
    '''
    Aims to improve the contrast and brightness of an input image using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    '''
    #Convert the input image from the BGR color space to the LAB color
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    #Split the LAB image into its individual channels (L, A, and B) using
    l, a, b = cv2.split(lab_image)

    #Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    #Apply the CLAHE object to the L channel
    cl = clahe.apply(l)

    #Merge the processed L channel (cl) with the original A and B channels
    limg = cv2.merge((cl, a, b))

    #Convert the processed LAB image back to the BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_image

def inpaint_missing_region(image, mask_radius=22, inpaint_radius=3):
    # Create a black mask of the same size as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw a white circle in the bottom right corner of the mask,
    # representing the missing region to be inpainted
    circle_center = (188, 211)
    cv2.circle(mask, circle_center, mask_radius, 255, -1)

    # Perform inpainting on the input image using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return inpainted_image

def unwarp_image(image):
    # Calculate the projective transformation matrix
    src_pts = np.float32([[25, 12], [230, 5], [237, 230], [25, 236]])
    dst_pts = np.float32([[0, 0], [250, 0], [250, 250], [0, 250]])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the projective transformation to the image
    unwrapped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    return unwrapped_image


def sharpen_image(image):

    intensity = -0.2
    # Apply a Gaussian blur to the input image
    blurred_image = cv2.GaussianBlur(image, (0, 0), 5)

    # Calculate the difference between the original and blurred image
    diff = cv2.subtract(image, blurred_image)

    # Add the difference, multiplied by the intensity, back to the original image
    sharpened_image = cv2.addWeighted(image, 1.0 + intensity, diff, intensity, 0)

    return sharpened_image


def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_name}")
            continue

        inpainted_img = inpaint_missing_region(img) 
        unwarp_img = unwarp_image(inpainted_img)
        denoised_img = denoise_image(unwarp_img )
        sharpen_img = sharpen_image(denoised_img)
        enhanced_img = enhance_contrast_and_brightness(sharpen_img )
        out_img =  enhanced_img
            

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, out_img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for classification.')
    parser.add_argument('input_folder', help='Path to the folder containing test images')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = 'Results'
    process_images(input_folder, output_folder)