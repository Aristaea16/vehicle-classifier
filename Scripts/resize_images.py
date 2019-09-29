"""
	This script takes in a path to a directory of images and resizes them based on WIDTH.
	Made for ByteMyAscii CEN4020 project Fall 2019.
"""

import os
import cv2
import matplotlib.pyplot as plt


WIDTH = 100

# Uses opencv to resize an image without distortion
def resize_image_no_distortion(image, width = None, height = None, inter = cv2.INTER_AREA):
    dimensions = None
    (original_height, original_width) = image.shape[:2]

    # Do nothing
    if width is None and height is None:
        return image
    else:
        # Calculate new dimensions
        temp = width / float(original_width)
        dimensions = (width, int(original_height * temp))

    # Resize the image with the proper dimensions
    resized = cv2.resize(image, dimensions, interpolation = inter)

    # return the resized image
    return resized

def resize_images(images_directory):
    # Open the image directory
    for image in os.listdir(images_directory):
        try:
            # Get next image
            image_path = os.path.join(images_directory, image)
            image_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            # Resize image to width of 100 and height to an equivalent ratio
            new_image = resize_image_no_distortion(image_array, WIDTH)
            #plt.imshow(new_image, cmap = "gray")
            #plt.show()

            # Save the new image
            cv2.imwrite(image_path, new_image)
        except Exception as e:
            print(e)
            exit(0)


# Main
print("This script will scale images to a default of width 100px.")
print("Height is also reduced to the correct ratio")

# Get the path to the directory containing the images
images_directory = input("Enter the path to your directory (Ex. /home/laurie/images): ")
resize_images(images_directory)

print("Success")



