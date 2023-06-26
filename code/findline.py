def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.sum(binary_warped[:, :], axis=0)
    lane_base = np.argmax(histogram)
    img1 = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
    num = 0
    for num in range(len(histogram) - 1):
        cv2.line(img1, (num, int(720 - histogram[num] / 200)), (num + 1, int(720 - histogram[num + 1] / 200)),
                 (255, 255, 0), 5)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 25
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            cv2.rectangle(img1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 255), 3)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            cv2.rectangle(img1, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 255), 3)

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

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]



    num = 0
    for num in range(len(ploty) - 1):
        cv2.line(img1, (int(left_fitx[num]), int(ploty[num])), (int(left_fitx[num + 1]), int(ploty[num + 1])),
                 (0, 0, 255), 20)
        cv2.line(img1, (int(right_fitx[num]), int(ploty[num])), (int(right_fitx[num + 1]), int(ploty[num + 1])),
                 (0, 0, 255), 20)


    vertices = np.array([[(int(left_fitx[0]), int(ploty[0])), (int(left_fitx[num - 1]), int(ploty[num - 1])),
                          (int(right_fitx[num - 1]), int(ploty[num - 1])), (int(right_fitx[0]), int(ploty[0]))]])
    cv2.fillPoly(img1, vertices, (0, 255, 0))

    aP = [0.0, 0.0]

    if (lane_base >= 640):
        LorR = -1.0  # Right
        plotx = right_fitx
        fit = right_fit

    else:
        LorR = 1.0  # Left
        plotx = left_fitx
        fit = left_fit

    aimLaneP = [int(plotx[len(ploty) // 2]), int(ploty[len(ploty) // 2])]
    img1 = cv2.circle(img1, (aimLaneP[0], aimLaneP[1]), 25, (255, 0, 0), -1)

    lanePk = (1 / (2 * fit[0])) * aimLaneP[0] - fit[1] / (2 * fit[0])
    k_ver = -1 / lanePk
    theta = math.atan(k_ver)
    aP[0] = aimLaneP[0] + math.cos(theta) * (LorR) * width_pers / 2
    aP[1] = aimLaneP[1] + math.sin(theta) * (LorR) * width_pers / 2

    img1 = cv2.circle(img1, (int(aP[0]), int(aP[1])), 25, (0, 0, 255), -1)
    return img1
