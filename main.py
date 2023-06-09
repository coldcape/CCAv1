# ---- Sources ---- #
# https://en.wikipedia.org/wiki/Connected-component_labeling#Two-pass
# https://www.youtube.com/watch?v=oXlwWbU8l2o
# https://youtu.be/hMIrQdX4BkE and https://www.udacity.com/course/introduction-to-computer-vision--ud810
# https://youtu.be/ticZclUYy88


# ---- Imports ---- #
import cv2 as cv
import numpy as np

# ---- Input image ---- #
# Load the input image and convert to grayscale
img = cv.imread('cc_input.png')
gray_scaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshold the grayscale image to obtain a binary image
_, thresh = cv.threshold(gray_scaled, 127, 255, cv.THRESH_BINARY_INV)

# Perform connected components analysis on the binary image
_, markers = cv.connectedComponents(thresh)

# First pass
# Initialize variables for first pass
labels = np.zeros_like(markers)
background = 0
next_label = 1
linked = []

# Iterate over each pixel in the image
for row in range(labels.shape[0]):
    for column in range(labels.shape[1]):
        # If pixel is not part of the background
        if markers[row, column] != background:
            # Find neighboring labels
            neighbors = []
            if row > 0 and labels[row - 1, column] != background:
                neighbors.append(labels[row - 1, column])
            if column > 0 and labels[row, column - 1] != background:
                neighbors.append(labels[row, column - 1])

            # Assign new label if no neighbors exist
            if not neighbors:
                labels[row, column] = next_label
                linked.append({next_label})
                next_label += 1
            # Assign existing label if one neighbor exists
            else:
                neighbors = [n for n in neighbors if n != background]
                if not neighbors:
                    labels[row, column] = next_label
                    linked.append({next_label})
                    next_label += 1
                # Merge neighboring labels if multiple neighbors exist
                else:
                    L = min(neighbors)
                    labels[row, column] = L
                    for l in neighbors:
                        if l != L:
                            linked[l - 1] = linked[l - 1].union(linked[L - 1])

# Second pass
# Iterate over each pixel in the image
for row in range(labels.shape[0]):
    for column in range(labels.shape[1]):
        # Replace label with smallest label in the equivalence set
        if labels[row, column] != background:
            labels[row, column] = min(linked[labels[row, column] - 1])

# Bounding boxes
# Initialize variables for bounding box computation
bounding_boxes = {}
# Iterate over each pixel in the image
for row in range(labels.shape[0]):
    for column in range(labels.shape[1]):
        label = labels[row, column]
        if label != background:
            # Update bounding box for label if necessary
            if label not in bounding_boxes:
                bounding_boxes[label] = [column, row, column, row]
            else:
                x_min, y_min, x_max, y_max = bounding_boxes[label]
                bounding_boxes[label] = [min(x_min, column), min(y_min, row), max(x_max, column), max(y_max, row)]

# Get average height and width
# Calculate average height and width of bounding boxes
heights = []
widths = []
for label, bbox in bounding_boxes.items():
    x_min, y_min, x_max, y_max = bbox
    heights.append(y_max - y_min)
    widths.append(x_max - x_min)

avg_height = sum(heights) / len(heights)
avg_width = sum(widths) / len(widths)

# Drawing boundary boxes on CC
# Initialize variables for drawing bounding boxes on image
img_output = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
