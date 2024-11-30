# Omer Garabagi, 322471145
# Omer Chernia, 318678620

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    threshold = 215
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

       # Reshape back to original image size
        quantized_img = img_compressed.reshape(img_size[0], img_size[1])
        
        # Threshold the quantized image to black and white
        binary_img = (quantized_img >= threshold).astype('uint8') * 255
        
        res.append(binary_img)

    return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 0:
		y_pos-=1
	return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target):

    # Compute target histogram and prepare signature
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    bin_indices = np.arange(256).astype(np.float32)
    target_signature = np.array(
        [[h, i] for h, i in zip(target_hist, bin_indices) if h > 0],
        dtype=np.float32
    )

        # Define sliding window size
    window_height, window_width = 15, 10

    # Get the dimensions of the source image
    image_height, image_width = src_image.shape[:2]

    # Fixed x positions from 30 to 40 (size 10 pixels)
    x_start = 30
    x_end = x_start + window_width  # This will be 40

    # Iterate over y positions from 100 to 330
    for y in range(100, 115):
        # Ensure the window does not exceed the image boundaries
        if y + window_height > image_height:
            break
        # Extract the window at the fixed x positions
        window = src_image[y:y + window_height, x_start:x_end]
        # Compute window histogram and prepare signature
        window_hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
        # Compute EMD
        emd = np.sum(np.abs(np.cumsum(target_hist) - np.cumsum(window_hist)))
        #print(f"EMD at position (x={x_start}, y={y}): {emd}")
        if emd < 260:
            return True
    return False


# Sections a, b
def subtask_f(image):
    list_amount=[]
    for i in range(10):
        list_amount.append(get_bar_height(image, i))
    return list_amount

def subtask_g(image,max_height):
    list_pixels = subtask_f(image)
    list_students = []
    for i in range (10):
        list_students.append(round(max_height*list_pixels[i]/max(list_pixels)))
    return list_students


images, names = read_dir('data')
numbers, _ = read_dir('numbers')
max_height_list=[]
for j in range(0,7):
    for i in range(9,-1,-1):
        if (compare_hist(images[j], numbers[i])) :
            max_height_list.append(i)

gray_images=quantization(images)

for k in range(0,7):
    res = ", ".join(map(str, subtask_g(gray_images[k],max_height_list[k])))
    print("Histogram ", names[k], " gave ", res)
# cv2.imshow(_[5], numbers[7])
# cv2.imshow(names[3], images[3]) 
cv2.waitKey(0)
# cv2.destroyAllWindows() 



exit()


# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.

# print(f'Histogram {names[id]} gave {heights}')
