import numpy as np
import cv2

# todo note to self: remove and rewrite
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

img = None

def colour_query_mouse_callback(event, x, y, flags, param):
    # records mouse events at postion (x,y) in the image window

    # left button click prints colour information at click location to stdout
    clicked = False
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        # print ("BGR colour @ position (%d,%d) = %s" % (x,y, ', '.join(str(i) for i in img[y,x])))
        print("Pos:{},{} B:{} G:{} R:{}".format(x, y, img[y, x, 0], img[y, x, 1], img[y, x, 2]))
        #cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), [60, 60, 60])


path_to_image = "../../BBBC010_v1_images/1649_1109_0003_Amp5-1_B_20070424_A01_w2_15ADF48D-C09E-47DE-B763-5BC479534681.tif"

main_window_name = "Hello THOMAS"
cv2.setMouseCallback(main_window_name,colour_query_mouse_callback);


img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

img = adjust_gamma(img, 6.5)
#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # todo rewrite maybe
#ret, th1 = cv2.threshold(img,140,150,cv2.THRESH_BINARY)

cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
keep_processing = True
while keep_processing:
    cv2.imshow(main_window_name, img) # th1

    key = cv2.waitKey(40) & 0xFF
    if (key == ord('x')):
        keep_processing = False
