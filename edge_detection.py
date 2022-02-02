import numpy as np
import matplotlib.pyplot as plt

# read the image
img = plt.imread('images/971686.jpg')
img = plt.imread('images/edge_girl.png')
img = plt.imread('images/rubix.jpg')

plt.imshow(img)
plt.show()

# convert image to grayscale as we don't need colour information
greyImg = img.mean(axis=2, keepdims=True)/255.0
greyImg = np.concatenate([greyImg]*3, axis=2)


# Sobel filters
vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

n,m,d = greyImg.shape
edges_img = np.zeros_like(greyImg)

# create 3x3 image box and filter the edges with sobel filters
# range is started from 3 as it's considered as boundary
for row in range(3, n-2):
    for col in range(3, m-2):
        local_pixels = greyImg[row-1:row+2, col-1:col+2, 0]
        
        # vertical and horizantal filtered pixels ranges from -4 to 4. So the division
        # normalises it to the range 0 to 1
        vertical_transformed_pixels = vertical_filter*local_pixels
        vertical_score = vertical_transformed_pixels.sum()/4
        
        horizontal_transformed_pixels = horizontal_filter*local_pixels
        horizontal_score = horizontal_transformed_pixels.sum()/4
        
        # the edge pixel is created by averaging out the vertical and horizontal score
        edge_score = (vertical_score**2 + horizontal_score**2)**.5
        # edge_score is multiplied by 3 to create 3 channels similar to how the image
        # was stored before
        edges_img[row, col] = [edge_score]*3
edges_img = edges_img/edges_img.max()

plt.imshow(edges_img)
plt.show()
