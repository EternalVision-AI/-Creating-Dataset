import os
import cv2

def convert_and_save_images(input_folder, output_folder, output_format='jpg'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(len(os.listdir(input_folder)))
    # List all files in the input folder
    count=204
    for filename in os.listdir(input_folder):
        # Read the image
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue  # Skip files that are not images

        # Perform any desired image processing here
        # For example, converting to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Construct the output filename
        base_filename = count
        output_filename = f"{base_filename}.{output_format}"
        output_filepath = os.path.join(output_folder, output_filename)

        # Save the image
        cv2.imwrite(output_filepath, img)
        count+=1

# Example usage
input_folder = '1-1'
output_folder = '1-1'
convert_and_save_images(input_folder, output_folder, output_format='jpg')
