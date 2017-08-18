import numpy as np
import cv2

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension4

# Define source points for perspective transformation
src_points = np.float32([[200,720],[453,547],[835,547],[1100,720]])
# Define 4 destination points for perspective transformation
dst_points = np.float32([[320,720],[320,576],[960,576],[960,720]])

def undistort(img,mtx,dist):
    '''
    Undistort an image given the camera calibration coefficients
    '''
    return cv2.undistort(img, mtx, dist, None, mtx)

def get_eagle_eye(img):
    '''
    Gets eagle eye perspective of the source points in the current image.
    '''
    img_size = (img.shape[1],img.shape[0])

    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Warp image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def get_inverse_transform(img):
    '''
    Warps the detected lane boundaries back onto the original image.
    '''
    img_size = (img.shape[1],img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped

def draw_points(img, points):
    '''
    Draws lines in an image. Used to represent perspective transform.
    '''
    temp = np.copy(img)
    temp = cv2.line(temp, tuple(points[0]), tuple(points[1]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[1]), tuple(points[2]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[2]), tuple(points[3]), color=(0,0,255), thickness=4)
    temp = cv2.line(temp, tuple(points[3]), tuple(points[0]), color=(0,0,255), thickness=4)
    return temp

def get_binary_image(img, l_thresh=(60, 255), v_thresh=(30, 100), y_thresh=(30, 255), kernel_size = 15):
    '''
    Applies different gradients to the current frame in order to detect road lanes.
    '''
    img = np.copy(img)

    # Convert to HSV color space and separate the v channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]

    channel = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    hls_l_binary = np.zeros_like(scaled_sobel)
    hls_l_binary[(scaled_sobel >= l_thresh[0]) & (scaled_sobel <= l_thresh[1])] = 1

    # Convert to YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float)
    y_channel = yuv[:,:,0]
    v_channel = yuv[:,:,2]

    channel = cv2.GaussianBlur(y_channel, (kernel_size, kernel_size), 0)
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    yuv_y_binary = np.zeros_like(scaled_sobel)
    yuv_y_binary[(scaled_sobel >= y_thresh[0]) & (scaled_sobel <= y_thresh[1])] = 1

    white = np.zeros_like(scaled_sobel)
    white[(hls_l_binary == 1) | (yuv_y_binary == 1)] = 1

    channel = cv2.GaussianBlur(v_channel, (kernel_size, kernel_size), 0)
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    yellow = np.zeros_like(scaled_sobel)
    yellow[(scaled_sobel >= v_thresh[0]) & (scaled_sobel <= v_thresh[1])] = 1

    combined = np.zeros_like(scaled_sobel)
    combined[(yellow == 1) | (white == 1)] = 1
    return combined

def process_frame(frame,mtx,dist):
    # Undistort image
    undistorted = undistort(frame,mtx,dist)

    # Threshold image with default values
    thresholded = get_binary_image(undistorted)

    # Change perspective to eagle-eye view
    top_down = get_eagle_eye(thresholded)

    return undistorted, thresholded, top_down

def find_lanes_hist(top_down):
    '''
    Finds the origin of the road lanes using a histogram.
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(top_down[int(top_down.shape[0]*4/5):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    margin = 200
    leftx_base = np.argmax(histogram[margin:midpoint-margin]) + margin
    rightx_base = np.argmax(histogram[midpoint+margin:top_down.shape[1]-margin]) + midpoint + margin

    return leftx_base, rightx_base


def get_poly_from_last(binary_warped, left_fit, right_fit, margin=100):
    '''
    Fits a polynomial to the detected lanes. Assumes that previous frame was properly detected and a fit was
    correctly estimated.
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
        return None, None, 0, 0
    else:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_curverad, right_curverad = get_curvature(leftx, lefty, rightx, righty)

        return left_fit, right_fit, left_curverad, right_curverad


def get_polynomial(top_down, leftx_base, rightx_base, nwindows = 5, margin=60, minpix = 100, debug=False):
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

    init_height_left = top_down.shape[0] - np.ceil(window_height / float(2))
    init_height_right = top_down.shape[0] - np.ceil(window_height / float(2))

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    temp_left_idx = []
    temp_left_idx.append(np.array([], np.int))
    temp_rigth_idx = []
    temp_rigth_idx.append(np.array([], np.int))

    max_empty_win = 5
    counter_left = 0
    counter_right = 0

    # Step through the windows one by one
    for window in range(nwindows*2):
        # Identify window boundaries in x and y (and right and left)
        win_y_low_left = int(init_height_left - np.ceil(window_height / float(2)))
        win_y_high_left = int(init_height_left + np.ceil(window_height / float(2)))
        win_y_low_right = int(init_height_right - np.ceil(window_height / float(2)))
        win_y_high_right = int(init_height_right + np.ceil(window_height / float(2)))

        win_xleft_low = int(leftx_current - margin)
        win_xleft_high = int(leftx_current + margin)
        win_xright_low = int(rightx_current - margin)
        win_xright_high = int(rightx_current + margin)

        if debug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low_left),(win_xleft_high,win_y_high_left),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low_right),(win_xright_high,win_y_high_right),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low_left) & (nonzeroy < win_y_high_left) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low_right) & (nonzeroy < win_y_high_right) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Center next window around a point predicted using current fit
        # 1. For left lane
        if counter_left < max_empty_win:
            num_new_points_left = np.setdiff1d(np.array(good_left_inds),np.unique(np.concatenate(temp_left_idx)))
            if num_new_points_left.size > minpix:
                temp_left_idx.append(good_left_inds)
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                counter_left = 0
            else:
                counter_left += 1

            if np.concatenate(temp_left_idx).size== 0:
                # No line was detected.
                return None, None, 0, 0, None

            temp_l_idx = np.unique(np.concatenate(temp_left_idx))
            leftx = nonzerox[temp_l_idx]
            lefty = nonzeroy[temp_l_idx]
            temp_left_fit_x = np.polyfit(lefty, leftx, 1)
            temp_left_fit_y = np.polyfit(leftx, lefty, 1)

            cut_x = int(temp_left_fit_x[0]*win_y_low_left + temp_left_fit_x[1])
            #cut_x = int(temp_left_fit_x[0]*win_y_low_left**2 + temp_left_fit_x[1]*win_y_low_left + temp_left_fit_x[2])
            if cut_x > win_xleft_high or cut_x < win_xleft_low:
                # Out of window
                if cut_x > win_xleft_high:
                    pos_y = int(temp_left_fit_y[0]*win_xleft_high + temp_left_fit_y[1])
                    #pos_y = int(temp_left_fit_y[0]*win_xleft_high**2 + temp_left_fit_y[1]*win_xleft_high + temp_left_fit_y[2])
                    pos_x = win_xleft_high
                else:
                    pos_y = int(temp_left_fit_y[0]*win_xleft_low + temp_left_fit_y[1])
                    #pos_y = int(temp_left_fit_y[0]*win_xleft_low**2 + temp_left_fit_y[1]*win_xleft_low + temp_left_fit_y[2])
                    pos_x = win_xleft_low

            else:
                pos_y = win_y_low_left
                pos_x = cut_x

            init_height_left = int(pos_y)
            leftx_current = int(pos_x)
        #leftx_current = int(temp_left_fit[0]*win_y_low**2 + temp_left_fit[1]*win_y_low + temp_left_fit[2])

        # 2. For right lane
        if counter_right < max_empty_win:
            num_new_points_right = np.setdiff1d(np.array(good_right_inds),np.unique(np.concatenate(temp_rigth_idx)))
            if num_new_points_right.size > minpix:
                temp_rigth_idx.append(good_right_inds)
                right_lane_inds.append(good_right_inds)
                counter_right = 0
            else:
                counter_right += 1

            if np.concatenate(temp_rigth_idx).size== 0:
                # No line was detected.
                return None, None, 0, 0, None

            temp_r_idx = np.unique(np.concatenate(temp_rigth_idx))
            rightx = nonzerox[temp_r_idx]
            righty = nonzeroy[temp_r_idx]
            temp_right_fit_x = np.polyfit(righty, rightx, 1)
            temp_right_fit_y = np.polyfit(rightx, righty, 1)

            cut_x = int(temp_right_fit_x[0]*win_y_low_right + temp_right_fit_x[1])
            #cut_x = int(temp_right_fit_x[0]*win_y_low_right**2 + temp_right_fit_x[1]*win_y_low_right + temp_right_fit_x[2])
            if cut_x > win_xright_high or cut_x < win_xright_low:
                # Out of window
                if cut_x > win_xright_high:
                    pos_y = int(temp_right_fit_y[0]*win_xright_high + temp_right_fit_y[1])
                    #pos_y = int(temp_right_fit_y[0]*win_xright_high**2 + temp_right_fit_y[1]*win_xright_high + temp_right_fit_y[2])
                    pos_x = win_xright_high
                else:
                    pos_y = int(temp_right_fit_y[0]*win_xright_low + temp_right_fit_y[1])
                    #pos_y = int(temp_right_fit_y[0]*win_xright_low**2 + temp_right_fit_y[1]*win_xright_low + temp_right_fit_y[2])
                    pos_x = win_xright_low

            else:
                pos_y = win_y_low_right
                pos_x = cut_x

            init_height_right = int(pos_y)
            rightx_current = int(pos_x)

        #rightx_current = int(temp_right_fit[0]*win_y_low**2 + temp_right_fit[1]*win_y_low + temp_right_fit[2])


    # Concatenate the arrays of indices
    left_lane_inds = np.unique(np.concatenate(left_lane_inds))
    right_lane_inds = np.unique(np.concatenate(right_lane_inds))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
        return None, None, 0, 0, None
    else:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if debug:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_curverad, right_curverad = get_curvature(leftx, lefty, rightx, righty)

        return left_fit, right_fit, left_curverad, right_curverad, out_img

def get_curvature(leftx, lefty, rightx, righty):
    '''
    Computes the curvature of both lanes in meters.
    '''
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

    if left_curverad < 10 or right_curverad < 10:
        return False

    # 1. Check that the horizontal distance is close to the estimated 3.7 meters.
    dist = (right_fitx - left_fitx) * xm_per_pix    # Horizontal distance between points (in meters)
    if np.min(dist) < 1.0:
        return False

    dist_bottom = (right_fitx[-1] - left_fitx[-1]) * xm_per_pix
    if dist_bottom > 4.2 or dist_bottom < 2.3:
        return False

    return True

def find_position(image_size, left_point, right_point):
    '''
    Finds the position of the car relative to the center
    '''
    middle_of_image = image_size/2

    center_of_road = (left_point + right_point)/2
    # Define conversions in x and y from pixels space to meters
    return (middle_of_image - center_of_road)*xm_per_pix
