import numpy as np
import cv2

original_image, working_image = None, None
relative_image_folder_path = "../../BBBC010_v1_images/"
image_names = []
current_image_index = -1
current_image_path = ""


def load_image(image_path):
    global working_image, original_image
    working_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = working_image[::]
    return


def load_image_paths():
    global image_names
    image_names_file = open("imgs.txt", "r")
    image_names = image_names_file.read().split("\n")
    return


def cycle_images():
    global current_image_index, current_image_path
    current_image_index = (current_image_index + 1) % len(image_names)
    current_image_path = relative_image_folder_path + image_names[current_image_index]
    load_image(current_image_path)
    return


def load_and_process_next_image():
    global working_image
    cycle_images()
    working_image = adjust_gamma(working_image, 6.5)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # todo rewrite?
    # ret, th1 = cv2.threshold(img,140,150,cv2.THRESH_BINARY)
    return


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values todo remove and rewrite?
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def mouse_press_callback(event, x, y, flags, param):
    global working_image
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pos:{},{} B:{} G:{} R:{}".format(x, y, working_image[y, x, 0], working_image[y, x, 1],
                                                working_image[y, x, 2]))


main_window_name = "n - cycle images, x - exit"
cv2.setMouseCallback(main_window_name, mouse_press_callback)
cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
load_image_paths()
cycle_images()
keep_processing = True

while keep_processing:
    original_and_working = np.hstack((original_image, working_image))
    cv2.imshow(main_window_name, original_and_working)

    key = cv2.waitKey(40) & 0xFF
    if key == ord('x'):
        keep_processing = False
    elif key == ord('n'):
        load_and_process_next_image()
