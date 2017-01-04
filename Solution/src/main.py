import numpy as np
import cv2

original_image_w1, working_image_w1 = None, None
original_image_w2, working_image_w2 = None, None
relative_image_folder_path = "../../BBBC010_v1_images/"
image_names = None
current_index = -1
current_w1_path = ""
current_w2_path = ""


def load_images(w1_path, w2_path):
    global working_image_w1, original_image_w1
    global working_image_w2, original_image_w2
    working_image_w1 = cv2.imread(w1_path, 0)  # 0 for gray scale
    original_image_w1 = working_image_w1[::]
    working_image_w2 = cv2.imread(w2_path, 0)  # 0 for gray scale
    original_image_w2 = working_image_w2[::]
    return


def load_image_paths():
    global image_names
    image_names_file = open("imgs.txt", "r")
    image_names = image_names_file.read().split("\n")
    return


def cycle_images():
    global current_index, current_w1_path, current_w2_path
    current_index = (current_index + 1) % len(image_names)
    current_w1_path = relative_image_folder_path + image_names[current_index]
    current_index = (current_index + 1) % len(image_names)
    current_w2_path = relative_image_folder_path + image_names[current_index]
    load_images(current_w1_path, current_w2_path)
    return


def load_and_process_next_images():
    global working_image_w1, working_image_w2
    cycle_images()
    working_image_w1 = process_image(working_image_w1)
    working_image_w2 = process_image(working_image_w2)
    return


def process_image(img):
    # img = adjust_gamma(img, 6.5)
    img = cv2.equalizeHist(img)
    # todo tasks from the .docx
    return img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values todo remove and rewrite?
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def mouse_press_callback(event, x, y, flags, param):
    global working_image_w1
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pos:{},{} B:{} G:{} R:{}".format(x, y, working_image_w1[y, x, 0], working_image_w1[y, x, 1],
                                                working_image_w1[y, x, 2]))


main_window_name = "n - cycle images, x - exit"
cv2.setMouseCallback(main_window_name, mouse_press_callback)
cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
load_image_paths()
cycle_images()
keep_processing = True

while keep_processing:
    w1 = np.hstack((original_image_w1, working_image_w1))
    w2 = np.hstack((original_image_w2, working_image_w2))
    both = np.vstack((w1, w2))
    cv2.imshow(main_window_name, both)

    key = cv2.waitKey(40) & 0xFF
    if key == ord('x'):
        keep_processing = False
    elif key == ord('n'):
        load_and_process_next_images()
