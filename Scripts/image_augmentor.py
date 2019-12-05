"""
	This script takes in a path to a directory of images and adds five random augmentations
	per image given. Made for ByteMyAscii CEN4020 project Fall 2019.
"""

import Augmentor

# Should be updated to the number of images you want to augment
num_images = 1908

# Path to the folder containing the base images to be augmented
p = Augmentor.Pipeline("/home/laurie/deep_learning/tensorflow/images_aug_test")

# Add five distorted images per base image
for i in range (0,5):
	p = Augmentor.Pipeline("/home/laurie/deep_learning/tensorflow/images_aug_test")

	# Add different distortions to all images with each distortion having a 50 percent
	#  chance of occurring
	p.flip_left_right(probability=0.5)
	p.random_contrast(probability=0.5, min_factor=0.5, max_factor=0.7)
	p.random_erasing(probability=0.5, rectangle_area=0.5)
	p.skew_tilt(probability=0.5, magnitude=0.3)
	p.greyscale(probability=0.5)

	p.status()
	p.sample(num_images)

print("Complete")


