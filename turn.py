import cv2
import numpy as np

# global variable
#pts1 = np.float32([[1100, 900], [600, 1225], [1375, 900], [1900, 1225]])

pts1=np.float32([[980, 900], [200, 1425], [1475, 900], [2300, 1425]]) #左上、左下、右上、右下
width,height=600,350
pts2=np.float32([[0,0],[0,height],[width,0],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)

def bird_eye(frame_in):
    frame=cv2.warpPerspective(frame_in,matrix,(width,height))
    #for x in range (0,4):
        #cv2.circle(frame_in,(pts1[x][0],pts1[x][1]),10,(0,0,255),cv2.FILLED)
    return frame
# Gradient with direction、Combine Color and Gradient Thresholding

def color_gradient_thresh(img, s_thresh=(170, 255), l_thresh=(30, 255), sx_thresh=(65, 100)):
    #鳥瞰圖邊緣檢測
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient and color
    color_gradient_binary = np.zeros_like(s_channel)
    color_gradient_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & (
                (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])) | (
                                      (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))] = 1
    return color_gradient_binary

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def getCurvatureForLanes(processed_img, prev_left_fitx, prev_right_fitx, prev_left_peak, prev_right_peak):
    # 得到左右兩車道的曲率、車道線點的 set
    yvals = []
    leftx = []
    rightx = []
    imageHeight = processed_img.shape[0]
    imageWidth = processed_img.shape[1]
    bufferForDecidingByDistanceFromMid = 10

    left_histogram = np.sum(processed_img[int(imageHeight / 4):, :int(imageWidth / 2)], axis=0)
    right_histogram = np.sum(processed_img[int(imageHeight / 4):, int(imageWidth / 2):], axis=0)

    # get local maxima
    starting_left_peak = np.argmax(left_histogram)
    leftx.append(starting_left_peak)

    starting_right_peak = np.argmax(right_histogram)
    rightx.append(starting_right_peak + imageWidth / 2)

    curH = imageHeight
    yvals.append(curH)
    increment = 25
    columnWidth = 150
    leftI = 0
    rightI = 0
    while (curH - increment >= imageHeight / 4):
        curH = curH - increment
        leftCenter = leftx[leftI]
        leftI += 1
        rightCenter = rightx[rightI]
        rightI += 1

        # calculate left and right index of each column
        leftColumnL = max((leftCenter - columnWidth / 2), 0)
        rightColumnL = min((leftCenter + columnWidth / 2), imageWidth)

        leftColumnR = max((rightCenter - columnWidth / 2), 0)
        rightColumnR = min((rightCenter + columnWidth / 2), imageWidth)

        # imageHeight/2 - (imageHeight - curH)
        lt1, lt2, lt3, lt4 = int(curH - increment), int(curH), int(leftColumnL), int(rightColumnL)
        rt1, rt2, rt3, rt4 = int(curH - increment), int(curH), int(leftColumnR), int(rightColumnR)
        leftHistogram = np.sum(processed_img[lt1:lt2, lt3:lt4], axis=0)
        rightHistogram = np.sum(processed_img[rt1:rt2, rt3:rt4], axis=0)

        left_peak = np.argmax(leftHistogram)
        right_peak = np.argmax(rightHistogram)
        if (left_peak):
            leftx.append(left_peak + leftColumnL)
        else:
            leftx.append(leftx[leftI - 1])

        if (right_peak):
            rightx.append(right_peak + leftColumnR)
        else:
            rightx.append(rightx[rightI - 1])
        yvals.append(curH)

    yvals = np.array(yvals)
    rightx = np.array(rightx)
    leftx = np.array(leftx)

    # Determine the fit in real space
    left_fit_cr = np.polyfit(yvals * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals * ym_per_pix, rightx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]

    return left_curverad, right_curverad, left_fitx, right_fitx, yvals, starting_right_peak, starting_left_peak

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    ceformed from `vertishe re`. Tst of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    # vertices 中數值恰好定義一多邊形，在 mask 中，這個多邊形的區域，用 255*k 這顏色填滿
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # left
    left_x = []
    left_y = []
    left_slope = []
    left_intercept = []

    # right
    right_x = []
    right_y = []
    right_slope = []
    right_intercept = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = cal_slope(x1, y1, x2, y2)
            if slope is not None and 0.5 < slope < 2.0:
                left_slope.append(cal_slope(x1, y1, x2, y2))
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
                left_intercept.append(y1 - x1 * cal_slope(x1, y1, x2, y2))
            if slope is not None and -2.0 < slope < -0.5:
                right_slope.append(cal_slope(x1, y1, x2, y2))
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
                right_intercept.append(y1 - x1 * cal_slope(x1, y1, x2, y2))
            # else continue
    # Line: y = ax + b
    # Calculate a & b by the two given line(right & left)

    # left
    if (len(left_x) != 0 and len(left_y) != 0 and len(left_slope) != 0 and len(left_intercept) != 0):
        average_left_x = sum(left_x) / len(left_x)
        average_left_y = sum(left_y) / len(left_y)
        average_left_slope = sum(left_slope) / len(left_slope)
        average_left_intercept = sum(left_intercept) / len(left_intercept)
        left_y_min = img.shape[0] * 0.6
        left_x_min = (left_y_min - average_left_intercept) / average_left_slope
        left_y_max = img.shape[0]
        left_x_max = (left_y_max - average_left_intercept) / average_left_slope
        cv2.line(img, (int(left_x_min), int(left_y_min)), (int(left_x_max), int(left_y_max)), color, thickness)

    # right
    if (len(right_x) != 0 and len(right_y) != 0 and len(right_slope) != 0 and len(right_intercept) != 0):
        average_right_x = sum(right_x) / len(right_x)
        average_right_y = sum(right_y) / len(right_y)
        average_right_slope = sum(right_slope) / len(right_slope)
        average_right_intercept = sum(right_intercept) / len(right_intercept)
        right_y_min = img.shape[0] * 0.6
        right_x_min = (right_y_min - average_right_intercept) / average_right_slope
        right_y_max = img.shape[0]
        right_x_max = (right_y_max - average_right_intercept) / average_right_slope
        cv2.line(img, (int(right_x_min), int(right_y_min)), (int(right_x_max), int(right_y_max)), color, thickness)
    if average_right_slope<1 and average_right_slope>0 and average_left_slope<1 and average_left_slope>0:
        print("turn right")
    if average_right_slope>(-1) and average_right_slope<0 and average_left_slope>(-1) and average_left_slope<0:
        print("turn left")


def cal_slope(x1, y1, x2, y2):
    if x2 == x1:  # devide by zero
        return None
    else:
        return ((y2 - y1) / (x2 - x1))


def intercept(x, y, slope):
    return y - x * slope


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)   ######
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def pipeline(image):
    # for filename in os.listdir(image):
    # path = os.path.join(image, filename)
    # image = mpimg.imread("test_images/solidYellowLeft.jpg")

    # 1. gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Smoothing
    blur_gray = cv2.GaussianBlur(image, (11, 11), 0)

    # 3. canny
    low_threshold = 75
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # cv2.imshow("windows", edges)
    # cv2.waitKey(0)

    # #4. masked
    imshape = image.shape
    vertices = np.array([[(0, 1432), (0, 1040), (729, 932), (1545, 947), (2267, 1432)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)################
    # cv2.imshow("windows", masked_edges)
    # cv2.waitKey(0)

    # 5. Hough transform 偵測直線
    # Hesse normal form
    rho = 2 # 原點到直線的最短直線距離
    theta = np.pi / 180 # 最短直線與X軸的夾角
    threshold = 100
    min_line_len = 25
    max_line_gap = 25
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)########

    test_images_output = weighted_img(line_img, image, α=0.8, β=1., γ=0.)########

    return test_images_output

#path = "./video_Trim.mp4"
path="./test.mp4"
cap = cv2.VideoCapture(path)
i = 0
while cap.isOpened():
    ret, frame = cap.read() #frame=原圖
    b_img=bird_eye(frame)
    w_img=color_gradient_thresh(b_img,s_thresh=(170, 255), l_thresh=(30, 255), sx_thresh=(65, 100))
    left_curverad, right_curverad, left_fitx, right_fitx, yvals, right_peak, left_peak = getCurvatureForLanes(w_img,[], [], [],[])
    #print(right_curverad)
    img=pipeline(frame)
    '''
    left="left_curverad:"+str(left_curverad)
    right="right_curverad:"+str(right_curverad)
    cv2.putText(img,left,(50,50),cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,255), 2)
    cv2.putText(img,right, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    '''

    if abs(left_curverad-right_curverad)<100:
        pred="straight"
    elif left_curverad>right_curverad:
        pred = "turn left"
    else:
        pred = "turn right"
    cv2.putText(img, pred, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    ###
    cv2.namedWindow("windows", 0)
    cv2.resizeWindow("windows", 700, 480)
    cv2.imshow("windows", img)
    #cv2.waitKey(1)q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
