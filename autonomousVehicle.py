# Video Expanation: https://youtu.be/Birvs5MYOLY?si=HuYnzLCldk9Db6qK
import cv2
import numpy as np

vidcap = cv2.VideoCapture("LaneVideo.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

prevLx = []
prevRx = []

while success:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640,480))

    ## Choosing points for perspective transformation
    tl = (222,387)
    bl = (70 ,472)
    tr = (400,380)
    br = (538,472)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        y -= 40
        
    # Ensure lx and rx are not empty
    if len(lx) == 0:
        lx = prevLx
    else:
        prevLx = lx
    if len(rx) == 0:
        rx = prevRx
    else:
        prevRx = rx

    # Ensure both lx and rx have the same length
    min_length = min(len(lx), len(rx))

    ## Autonomous Vehicle
    # Create a list of points for the left and right lane coordinates
    left_points = [(lx[i], y + i * 40) for i in range(min_length)]
    right_points = [(rx[i], y + i * 40) for i in range(min_length)]

    # Fit a second-order polynomial to the lane points
    left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
    right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)

    # Calculate the curvature
    y_eval = 480
    left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
    right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
    curvature = (left_curvature + right_curvature) / 2

    # Calculate the lane offset
    lane_center = (left_base + right_base) / 2
    car_position = 320  # Assuming the car is centered in the image
    lane_offset = (car_position - lane_center) * 3.7 / 640  # Convert to meters

    # Calculate the steering angle
    # Assuming the camera is mounted at the center of the car
    steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi

    # Calculate the end point of the line based on the angle
    line_length = 100  # Length of the line
    end_x = int(320 + line_length * np.sin(np.radians(steering_angle)))
    end_y = int(480 - line_length * np.cos(np.radians(steering_angle)))

    # Create the top and bottom points for the quadrilateral
    top_left = (lx[0], 472)
    bottom_left = (lx[min_length-1], 0)
    top_right = (rx[0], 472)
    bottom_right = (rx[min_length-1], 0)
    
    # Define the quadrilateral points
    quad_points = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)

    # Reshape quad_points to the required shape for fillPoly
    quad_points = quad_points.reshape((-1, 1, 2))

    # Create a copy of the transformed frame
    overlay = transformed_frame.copy()

    # Draw the filled polygon on the transformed frame
    cv2.fillPoly(overlay, [quad_points], (0, 255, 0))

    alpha = 0.2 # Opacity factor
    cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

    # Display the transformed frame with the highlighted lane
    cv2.imshow("Transformed Frame with Highlighted Lane", overlay)

    # Inverse perspective transformation to map the lanes back to the original image
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    original_perpective_lane_image = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))

    # Combine the original frame with the lane image
    result = cv2.addWeighted(frame, 1, original_perpective_lane_image, 0.5, 0)

    # Draw a straight line in the center of the frame pointing with the current angle
    cv2.line(result, (320, 480), (end_x, end_y), (255, 0, 0), 2)

    # Display the curvature, offset, and angle on the frame
    cv2.putText(result, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow('Lane Detection', result)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
