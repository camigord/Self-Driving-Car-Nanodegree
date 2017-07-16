# Finding Lane Lines on the Road

First project in the UDACITYs Self-Driving Car Nanodegree.

## Overview:

_"When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm."_

## Detection pipeline:

The proposed pipeline consists of 7 steps:
  1. The images are converted to grayscale.
  2. The images are blurred by applying gaussian smoothing.
  3. I use Canny edge detector to identify edges on the images based on the defined thresholds.
  4. A region of interest is defined as a trapezoid which is more or less centered at the position where the road lanes are expected to be. A mask is defined based on this ROI and we keep only the pixels inside the trapezoid. 
  5. I use the Hough transformation to identify lines inside the ROI. 
  6. The detected lines are then filtered, averaged and extrapolated to map out the full extent of the left and right lane boundaries. 
      - I filter the lines based on their slope in order to create two groups (left lane = negative slope; right lane = positive slope).
      - Both the slope and the intersection point 'b' of each line are calculated.
      - I average (I actually use the median to minimize the effect of outliers) the slope and the intersection points of the lines belonging to each of the two groups. 
      - Having these new 'averaged' slopes and intersections, I compute the points at the boundaries defined by the ROI.
      - Now I can just plot these two lines on top of the image.
  7. The detected lanes are stacked on top of the original image.
  
  
 Additional information can be found in the project [writeup](writeup.md)

<table>
  <tr>
    <td><img src="./assets/solidYellowLeft.gif?raw=true" width="400"></td>
    <td><img src="./assets/challenge.gif?raw=true" width="400"></td>
  </tr>
</table>
