# Finding Lane Lines on the Road

<table>
  <tr>
    <td><img src="./assets/solidYellowLeft.gif?raw=true" width="400"></td>
    <td><img src="./assets/challenge.gif?raw=true" width="400"></td>
  </tr>
</table>

## Overview:

This is my implementation code for the fisrt  project in the UDACITY Self-Driving Car Nanodegree.

<blockquote>
     <p>
     _"When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm."_
     </p>
</blockquote>

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

  The images below show an example of the lines detected after step 5 (left) and after filtering and extrapolating those lines in step 6 (right).

<img src="./test_images_output/solidWhiteCurve.jpg" width="350" height="200" /> <img src="./test_images_output/improved_solidWhiteCurve.jpg" width="350" height="200" />  

## Potential shortcomings with the current pipeline

There are several potential shortcomings, mainly because we are hardcoding most of the pipeline...
  - The thresholds for the Canny edge detector need to be tuned according to the image. Illumination conditions, shadows etc. can have a significant effect on the performance.
  - The ROI is probably the most hardcoded step in the pipeline. What if, for any reason, the camera moves a little bit during driving? In this case, the ROI is no longer valid and the detection performance could be drastically compromised. Moreover, a ROI does not guarantee that there won't be any distractors during driving. As seen in the 'Challenge' video, shadows or patches on the road located inside the ROI may be detected as lanes and could have potentially dangerous consequences  during driving.  
  - When extrapolating the detected lines, outliers could have an strong effect on the final result. You could decrease the amount of outliers by tuning the different parameters (like the ones in the Hough transformation) but doing that would very likely also decrease the sensitivity of the final detector. Lower sensitivity means that the system won't be able to detect any lines in some particular situations.

### 3. Possible improvements

One possible improvement would involve the way the detected lines are filtered. Right now, after applying the Hough transformation, we use all the detected lines and categorize them according to their slope. Because we know the location of our camera and because we are looking for road lanes, we could filter out any line which slope is not within a given range. For example an horizontal line is clearly not a road lane (at least not the kind we are looking for right now). Filtering out these lines will improve the accuracy and robustness of the final detector and would solve some of the problems I observed in the 'Challenge' video, where patches on the road and shadows were detected as almost horizontal lines.

A second improvement would be to keep a moving average of the _K_ last detected lines. This average would be updated once per frame after the lanes have been detected and we could even constraint these updates if we believe that the quality of the current detection is not good enough. Keeping these moving average would help us dealing with outliers and with particularly challenging scenes.

Another possible, although more complex improvement, could be to train a model to automatically tune the pipeline parameters based on the current image. This could be done either by manually tuning the parameters under different illumination and/or environmental conditions and by training a model to predict which parameters would work better given the current image statistics, or by defining some "_performance metric_" and by automatically learning to optimize the parameters in order to maximize performance.

Leaving the pipeline aside, however, it would be possible to improve the performance and robustness of the detector by training a deep model. Although a large amount of training samples and _labels_ would be required, the performance of such a model would be significantly better.
