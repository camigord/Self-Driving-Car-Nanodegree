import numpy as np
import cv2

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension4

def undistort(img,mtx,dist):
    '''
    Undistort an image given the camera calibration coefficients
    '''
    return cv2.undistort(img, mtx, dist, None, mtx)

def get_eagle_eye(img, src):
    '''
    Gets eagle eye perspective of the source points in the current image. The destination points
    are calculated based on offset
    '''
    img_size = (img.shape[1],img.shape[0])

    # Define 4 destination points
    #dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])
    dst = np.float32([[320,720],[320,576],[960,576],[960,720]])

    w,h = 1280,720
    x,y = 0.5*w, 0.8*h
    # Define 4 destination points
    dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])


    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, dst

def get_inverse_transform(img, src, dest):
    img_size = (img.shape[1],img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

def draw_points(img, points):
    temp = np.copy(img)
    temp = cv2.line(temp, tuple(points[0]), tuple(points[1]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[1]), tuple(points[2]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[2]), tuple(points[3]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[3]), tuple(points[0]), color=(0,0,255), thickness=4)
    return temp

def get_binary_image(img, s_thresh=(170, 255), sx_thresh=(20, 100), h_thresh=(20,30)):
    img = np.copy(img)

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    h_channel = hsv[:,:,0]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold hue channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # color_binary = np.dstack(( h_binary, sxbinary, s_binary))
    # Combine binaries
    combined = np.zeros_like(scaled_sobel)
    combined[(h_binary == 1) | (sxbinary == 1) | (s_binary == 1)] = 1
    return combined

def process_frame(frame,mtx,dist):
    # Undistort image
    undistorted = undistort(frame,mtx,dist)
    # Threshold image with default values
    thresholded = get_binary_image(undistorted)

    # Define source points for perspective transformation
    # src_points = np.float32([[596,447],[685,447],[1115,720],[196,720]])
    src_points = np.float32([[200,720],[453,547],[835,547],[1100,720]])

    # Change perspective to eagle-eye view
    top_down, dst_points = get_eagle_eye(thresholded, src_points)

    return top_down

def find_lanes_hist(top_down):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(top_down[int(top_down.shape[0]/2):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base

def get_polynomial(top_down, leftx_base, rightx_base, nwindows = 9, margin=100, minpix = 50, debug=True):
    '''
    Implementing a sliding window approach to detect points in left and right lanes and fit a second order polynomial
        nwindows is the number of sliding windows
        margin defines the width of the windows
        minpix sets the minimum number of pixels to be found to recenter window
    '''
    # Set height of windows
    window_height = np.int(top_down.shape[0]/nwindows)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((top_down, top_down, top_down))*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = top_down.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = top_down.shape[0] - (window+1)*window_height
        win_y_high = top_down.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if debug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if debug:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_curverad, right_curverad = get_curvature(leftx, lefty, rightx, righty)

    return left_fit, right_fit, left_curverad, right_curverad, out_img

def get_curvature(leftx, lefty, rightx, righty):
    y_eval_left = 700.
    y_eval_right = 700.

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def sanity_check(left_fitx, right_fitx, left_curverad, right_curverad):
    # 1. Check that curvature is within the same order of magnitude.
    if left_curverad < 300 or left_curverad > 700 or right_curverad < 300 or right_curverad > 700:
        return False

    # 2. Check that the horizontal distance is close to the estimated 3.7 meters.
    # I assume lanes are parallel if the distance between all horizontal points is similar.
    dist = (right_fitx - left_fitx) * xm_per_pix    # Horizontal distance between points (in meters)
    m = np.mean(dist)
    std = np.std(dist)
    if m+std > 4.0 or m-std < 3.4:  # If distance remains within given range [3.4, 4.0]
        return False

    return True
