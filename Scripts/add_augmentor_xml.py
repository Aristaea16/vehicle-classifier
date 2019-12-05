"""
	This script takes in a directory of augmented images and a directory of
	the base images with the xml from LabelImg and generates more xml for the
	augmented images to remove the need to hand-label them. Made for ByteMyAscii 
	CEN4020 project Fall 2019.
"""

import os
from shutil import copyfile
import xml.etree.ElementTree as ET

# Update the two tags containing the image path in each file
def update_xml_tags(file_path, image):
    # Name of the image file without the path
    image_name = image.rsplit('/', 1)[1]

    tree = ET.parse(file_path)
    root = tree.getroot()

    tag = root.findall("filename")
    tag[0].text = image_name

    tag = root.findall("path")
    tag[0].text = "/home/laurie/deep_learning/tensorflow/images/" + image_name

    tree.write(file_path)

# Get all the images
images = []
for file in os.listdir("/home/laurie/deep_learning/tensorflow/images_aug_test/output"):
    if file.endswith(".jpg"):
        images.append(os.path.join("/home/laurie/deep_learning/tensorflow/images_aug_test/output", file))

# Get all the xml files
for file in os.listdir("/home/laurie/deep_learning/tensorflow/images_aug_test"):
    if file.endswith(".xml"):
        file_path = os.path.join("/home/laurie/deep_learning/tensorflow/images_aug_test", file)
        print("file is: " + str(file))

        # For naming new xml files
        counter = 1

        # Create a new xml file for each augmented image
        for image in images:
            if image.find(file[:-4]) > 0:
                print(image)

                # Create a new copy of the xml file
                image_name = str(file_path[:-4] + "_" + str(counter) + ".xml")
                new_file_path = os.path.join("/home/laurie/deep_learning/tensorflow/images_aug_test", image_name)
                copyfile(file_path, new_file_path)

                update_xml_tags(new_file_path, image)

                counter += 1


