# **Finding Lane Lines on the Road** 
---
**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/improved_solidWhiteCurve.jpg "Solid Lane"
[image2]: ./test_images_output/solidWhiteCurve.jpg "Dash Lane"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

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

<img src="./test_images_output/solidWhiteCurve.jpg" width="350" height="200" /> <img src="./test_images_output/improved_solidWhiteCurve.jpg" width="350" height="200" />  

### 2. Identify potential shortcomings with your current pipeline

There are several potential shortcomings, mainly because we are hardcoding most of the pipeline...
  - The thresholds for the Canny edge detector need to be tuned according to the image. Illumination conditions, shadows etc. can have a significant effect on the performance.
  - The ROI is probably the most hardcoded step in the pipeline. What if, for any reason, the camera moves a little bit during driving? In this case, the ROI is no longer valid and the detection performance could be drastically compromised. Moreover, a ROI does not guarantee that there won't be any distractors during driving. As seen in the 'Challenge' video, shadows or patches on the road located inside the ROI may be detected as lanes and could have potentially dangerous consequences  during driving.  
  - When extrapolating the detected lines, outliers could have an strong effect on the final result. You could decrease the amount of outliers by tuning the different parameters (like the ones in the Hough transformation) but doing that would very likely also decrease the sensitivity of the final detector. Lower sensitivity means that the system won't be able to detect any lines in some particular situations.

### 3. Suggest possible improvements to your pipeline

One possible improvement would involve the way the detected lines are filtered. Right now, after applying the Hough transformation, we use all the detected lines and categorize them according to their slope. Because we know the location of our camera and because we are looking for road lanes, we could filter out any line which slope is not within a given range. For example an horizontal line is clearly not a road lane (at least not the kind we are looking for right now). Filtering out these lines will improve the accuracy and robustness of the final detector and would solve some of the problems I observed in the 'Challenge' video, where patches on the road and shadows were detected as almost horizontal lines.

Another possible, although more complex improvement, could be to train a model to automatically tune the pipeline parameters based on the current image. This could be done either by manually tuning the parameters under different illumination and/or environmental conditions and by training a model to predict which parameters would work better given the current image statistics, or by defining some "_performance metric_" and by automatically learning to optimize the parameters in order to maximize performance. 

Leaving the pipeline aside, however, it would be possible to improve the performance and robustness of the detector by training a deep model. Although a large amount of training samples and _labels_ would be required, the performance of such a model would be significantly better.
