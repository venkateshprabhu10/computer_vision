# Import the necessary libraries
import argparse
import cv2
import imutils

# Build the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input image")
ap.add_argument("-o", "--output", help="Path to output")

args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["input"])

# Convert to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.blur(gray, (2, 2))

# Adaptive Thresholding the image to create a mask
thresh = cv2.adaptiveThreshold(
    blurred, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -5)

# Find and grab the contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Draw boundary for the found objects
for c in cnts:
    cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

# Print the total objects found
if len(cnts) > 0:
    cv2.putText(image, f"Objects found: {len(cnts)}", (20, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

# Write the image to local disk
op = args["output"] if args["output"] else "final.png"
cv2.imwrite(op, image)
