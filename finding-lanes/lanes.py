import cv2
import numpy as np
import matplotlib.pyplot as plt

## 0. LOADING THE IMAGE


def loadImage(path):
    image = cv2.imread(path)
    return image

img = loadImage('road_image.jpg')
lane_image = np.copy(img)

## 1. GRAY SCALING
'''
 Processing a single channel is faster than processing a
 three channel image that's why grayscale is needed and less
 computationally intensive.
'''

def grayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray = grayScale(lane_image)



## 2. GAUSSION BLUR : to remove noise from the image.
'''
 A Gaussian filter is a linear filter. It's usually used to blur
 the image or to reduce noise. If you use two of them and subtract,
 you can use them for "unsharp masking" (edge detection).
 The Gaussian filter alone will blur edges and reduce contrast.
'''

def blurImage(image):
    return cv2.GaussianBlur(image, (5,5), 0)

blur = blurImage(gray)


## 3. CANNY function : To detect edges.

'''
We notice the changes in intensity when there are sharp changes in
the values of pixel in an image. e.g. from 0 to 255 in the next
column. It is used to detect edges in the image.

What operator can be used to detect the rapid changes in brightness
in our image?
So how can we find the sharp changes in intensity in all directions
in our image?

So, we'll use Canny function which will take derivative on both x
and y and will major adjacent changes in intensity in all directions
x and y. So, it'll check how pixels are changing in all directions
and hence will know when the brightness is high and low for pixels.

Canny traces the edges with large changes in intensity(large gradient)
in an outline of white pixels.

If the gradient is upper than high_threshold then it is accepted
as an edge pixel and if it is lower than the low_threshold then
it is rejected. and if it is between the both then it is accepted
only if it is connected to a strong edge.
'''

def canny(image):

    gray = grayScale(image)
    blur = blurImage(gray)
    canny = cv2.Canny(blur, 50, 150)

    return canny

canny_image = canny(lane_image)

## 4. : Finding lane lines

'''
So, here we'll define the area of our interest where we want
the cv to show the edges.
'''

def area_of_interest(image):

    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height),(550, 250)]])

    mask = np.zeros_like(image)

    '''
    ill the mask with the polygons in our case it's the triangle
    '''
    cv2.fillPoly(mask, polygons, 255) ## 255 is for white pixels

    '''
    With the help of mask we'll try to use only a certain area to
    focus.

    And now we'll use the '&' operator on the canny image which
    detected the edges and the mask which shows us the area of
    interest. '&' operator will give us the area of interest in white
    pixels and other pixels will have zero values.

    '''

    masked_image = cv2.bitwise_and(image,mask)

    return masked_image

cropped_image = area_of_interest(canny_image)


## 5. HOUGH TRANSFORM

'''
Two pixels with a single radian, threshold number of intersections
for the line to be accepted
'''


def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
        minLineLength = 40, maxLineGap = 5)

line_image = display_line(lane_image, lines)

'''
So, this line_image will show us the blue lines of the path
on a black path. So, we have to put this on our original image.
when we'll add both images we'll give 0.8 weight to the lane_image
and 1 to the line_image so that lines are more visible on the
surface of the lane_image with less brightness. And the last values
is the gamma value and we're putting it a scalar 1.
'''

final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('final_image', final_image)
cv2.waitKey(0)
