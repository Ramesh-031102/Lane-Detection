import numpy as np  
import cv2

def img2edge(img):

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_image  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define ranges for 'yellow', pixels within this range will be picked

    lower_yellow = np.array([25, 50, 70])
    upper_yellow = np.array([35, 255, 255])

    # cv2.inRange(): Picks pixels from the image that are in the specified range
    
    mask_y = cv2.inRange(hsv_image, lower_yellow, upper_yellow)      
    mask_w = cv2.inRange(gray_image, 216, 255)
  
    # Compute Bitwise OR, combining both the white and yellow pixels

    mask_yw = cv2.bitwise_or(mask_y, mask_w)

    # Compute Bitwise AND of mask_yw with gray_img, pixels that were yellow or 
    # white will have the same intensity as the original grayscale image, the 
    # other pixels will be removed.

    mask_yw_image = cv2.bitwise_and(mask_yw, gray_image) 

    img_blur = cv2.GaussianBlur(mask_yw_image, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 70, 200)

    return img_canny
    
def roi_select(img, canny):

    # Define the vertices of the region of interest
         
    height, width, _ = img.shape
    lower_left = [0,0.95*height]
    lower_right = [width,height]
    top_left = [0.375*width,0.375*height]
    top_right = [0.625*width,0.375*height]
    vertices = np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)
    
    mask = np.zeros_like(canny)                                      # creates a numpy array of the same dimensions as img
    fill_color = 255                                                 # parameter for cv2.fillPoly function
    cv2.fillPoly(mask, pts = [vertices], color = fill_color)         # pixels within 'vertices' in 'mask' will be made WHITE while all other pixels will be BLACK

    return cv2.bitwise_and(canny, mask)

def draw_lines(canny_roi, rho_acc, theta_acc, thresh, minLL, maxLG):
    
    line_img = np.zeros_like(canny_roi)
    lines = cv2.HoughLinesP(canny_roi,rho_acc,theta_acc,thresh,minLL,maxLG)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_img,(x1,y1),(x2,y2),(255,255,255),5)

    return line_img

def add_weighted(img, line_img):
    bgr_img = cv2.cvtColor(line_img,cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.8, bgr_img, 1, 0)

img = cv2.imread('LD_test_imgs/test_img01.jpeg')                  #Read the input image from the directory.
edge_img = img2edge(img)
roi_img = roi_select(img, edge_img)
hough_img = draw_lines(roi_img, 1, np.pi/180, 5, 5, 1)            #Change the parameters thresh, minLL, maxLG to get more accurate lines
lane_img = add_weighted(img, hough_img)
cv2.imshow('Lanes Detected',lane_img)

cv2.waitKey(0)
cv2.destroyAllWindows()