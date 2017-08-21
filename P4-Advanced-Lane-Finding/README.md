# Advanced Lane Finding


## Overview

Fourth project in the UDACITY Self-Driving Car NanoDegree: Identifying lane boundaries in a video from a front-facing camera on a car.

## Files

The current repository includes the following files:

* [Generate_video.ipynb](Generate_video.ipynb) containing the final pipeline and capable of loading and processing frames from the given videos.
* [Binary_Images.ipynb](Binary_Images.ipynb) describing how the binary images are generated and showing some filter examples.
* [project.ipynb](project.ipynb) Describing each step in the pipeline.
* [utils.py](./utils/utils.py) containing all the described functions.
* [Line.py](./utils/Line.py) containing the Line Class used to keep track of each lane (left & right lanes).

The generated videos are stored in the folder [output_videos](./output_videos).

## Approach

### 1. Camera calibration

I started the project by computing the camera calibration matrix and distortion coefficients using the images in [this](./camera_cal) folder. The entire code can be found [here](project.ipynb).

I used the OpenCV function _cv2.findChessboardCorners()_ to detect all the inner corners on each calibration image. Using these points together with a set of fixed _object points_ I am able to compute the camera calibration and distortion coefficients using the _cv2.calibrateCamera()_ function. Finally, I save the results as a pickle [file](./camera_cal/coefficients.p) in order to be able to load the coefficients any time when required.

The images below show the result of correcting the distortion on an image using the function _cv2.undistort()_.

<img src="./output_images/undistort.jpg">


### 2. Creating a thresholded binary image

The next step is to define a set of filters capable of thresholding the images in such a way that detecting the road lanes becomes easier.

The main idea is to find a color space in which the road lanes are clearly visible and in which the illumination conditions do not have a strong effect in the final outcome. As described in [this](Binary_Images.ipynb) notebook, I evaluated several color spaces (i.e HLS, YUV, HSV) while trying to find a combination of filters and parameters that perform well and provide robust results in different test scenarios. This was one of the most time consuming steps in this project and it was also a bit frustrating because a set of parameters which work perfectly fine for a set of images may produce a terrible result when tested on a different video sequence.

__Note__: Although I was able to find a set of filters which work perfectly fine for the _project video_, this pipeline worked very poorly on the other and more challenging videos. For this reason, I decided to use a more simple set of filters which happen to perform very well on the first video and very decently on the other two.

The final pipeline can be found in [utils.py](./utils/utils.py) with the name _get_binary_image(...)_ (lines 50 to 92). The function takes an image as an input and applies 3 different binary thresholds in two different color spaces (i.e HLS and YUV) in order to detect both yellow lanes and white lanes. The three resulting binary images are then combined in order to produce a final thresholded image. The table below shows the results after applying the corresponding filters for detecting the yellow and white lanes.

<table>
  <tr>
    <td align="center">Original</td>
    <td align="center">Yellow Lane</td>
    <td align="center">White Lane</td>
    <td align="center">Final Result</td>
  </tr>
  <tr>
    <td><img src="./output_images/test_image.jpg" width="500"></td>
    <td><img src="./output_images/yellow_lane.jpg" width="500"></td>
    <td><img src="./output_images/white_lane.jpg" width="500"></td>
    <td><img src="./output_images/combined.jpg" width="500"></td>
  </tr>
</table>

The images below show additional examples taken from the _challenge_video_ and the _harder_challenge_video_ respectively.

<img src="./output_images/binary1.jpg">

<img src="./output_images/binary2.jpg">

### 3. Perspective transformation

In order to make it easier for us to estimate the curvature of the road we are going to perform a perspective transformation to obtain an eagle-eye view of the scene. The function in charge of performing this transformation is called _get_eagle_eye(...)_ and appears in lines 18 through 28 of the file [utils.py](./utils/utils.py). The function takes as input an image and applies a perspective transformation using the function _cv2.getPerspectiveTransform()_ and a set of hardcoded _source_ and _destination_ points which are defined as:

```
src_points = np.float32([[200,720],[453,547],[835,547],[1100,720]])
dst_points = np.float32([[320,720],[320,576],[960,576],[960,720]])
```

The outcome of the function together with a representation of the source and destination points respectively can be seen in the images below.

<img src="./output_images/top_down_view.jpg">

The example above corresponds to a straight line and we can see that the result represents two straight and parallel lines. We can call this function with a different example and obtain the representation of a curve:

<img src="./output_images/top_down_view2.jpg">


### 4. Detecting lane pixels and fitting a polynomial


#### Peaks in a Histogram

The first step to identify the lane pixels is to recognize the origin of the lanes at the bottom of each image. After thresholding and applying the perspective transformation we end up with an image in which the lanes stand out clearly (_at least in most of the cases_). We can now compute a histogram along the columns in the lower part of the image like this:

```
histogram = np.sum(image[int(image.shape[0]*4/5):,:], axis=0)
```

I am only considering the lowest fifth part of the image when computing the histogram. The result looks something like this:

<img src="./output_images/histogram.jpg">

Once we have the histogram we can find the peaks at the left and right sides of the image. I decided to introduce a _margin  value_ in order to constraint the search regions both at the left and the right sides. The final function is called _find_lanes_hist(top_down)_ and appears in lines 106 through 119 of the file [utils.py](./utils/utils.py). The function takes the thresholded and warped top-down view image and return the origin of the left and the right lanes.

#### Sliding Window



### 5. Determine lane curvature and vehicle position

#### Radius of curvature

Now that we have a second order polynomial describing both left and right lanes we can estimate the _radius of curvature_ of each lane. The radius of curvature for any function _x=f(y)_ is given by:

<img src="./assets/equation.jpg">

Because our polynomial has the form f(y) = Ay<sup>2</sup> + By + C, this formula becomes:

<img src="./assets/equation2.jpg">

where A and B are the polynomial coefficients and _y_ represents the point along the curve where we want to estimate the radius of curvature.

Given that we want to estimate the radius of curvature in a metric unit, we need to convert the pixel space into real world space. To do this, we introduce two scaling factor: _ym_per_pix_ and _xm_per_pix_ which represent the length in meters that each pixel in the image represents. These values are defined based on a rough estimate which assumes that we are projecting a lane about 30 meters long and 3.7 meters wide:

```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

The function _get_curvature(...)_ in charge of computing the radius of curvature is located in lines 323 through 337 of the file [utils.py](./utils/utils.py). It takes the x and y position of the pixels detected on each lane as input and returns the curvature of each lane.

#### Distance to center

If we assume that the camera is mounted at the center of the car we can also estimate our distance to the center of the lane. The function _find_position(...)_ in lines 355 through 363 of the file [utils.py](./utils/utils.py) takes the position of the left and right lanes at the bottom of the image and returns the distance from the car to the center of the lane. To do this, we compute the center of the lane using the two previously mentioned points and compare this center with the center of the image. The difference is finally transformed into metric space.

Both the curvature and the distance to center are then drawn on top of the original image using the function _cv2.putText()_ in order to provide some feedback about the road.

### 6. Warping the detected lanes boundaries back onto the original image

<img src="./output_images/result.jpg">
