import numpy as np
import cv2

original_image_w1, working_image_w1 = None, None
original_image_w2, working_image_w2 = None, None
relative_image_folder_path = "../../BBBC010_v1_images/"
relative_image_output_folder_path = "../../img_out/"
original_file_name = ""
image_names = None
current_index = -1
current_w1_path = ""
current_w2_path = ""


def load_images(w1_path, w2_path):
    global working_image_w1, original_image_w1
    global working_image_w2, original_image_w2

    working_image_w1 = cv2.imread(w1_path, 0)  # 0 for gray scale
    original_image_w1 = working_image_w1.copy()

    working_image_w2 = cv2.imread(w2_path, 0)  # 0 for gray scale
    original_image_w2 = working_image_w2.copy()
    return


def load_image_paths():
    image_names_file = open("imgs.txt", "r")
    return image_names_file.read().split("\n")


def cycle_images():
    global current_index, current_w1_path, current_w2_path, original_file_name
    # Not ALL images have a w1 AND a w2 version - D24 for example has no w2, and so is EXCLUDED from imgs.txt
    current_index = (current_index + 1) % len(image_names)
    current_w1_path = relative_image_folder_path + image_names[current_index]
    current_index = (current_index + 1) % len(image_names)
    current_w2_path = relative_image_folder_path + image_names[current_index]
    original_file_name = str(image_names[current_index])
    load_images(current_w1_path, current_w2_path)
    return


def save_images(w1, w2):
    global original_file_name
    file_id = original_file_name[-47:-43]
    cv2.imwrite(relative_image_output_folder_path + file_id + "w1.jpg", w1)
    cv2.imwrite(relative_image_output_folder_path + file_id + "w2.jpg", w2)
    print("Saved {0}w1.jpg and {0}w2.jpg".format(file_id))
    pass


def load_and_process_next_images():
    global working_image_w1, working_image_w2
    cycle_images()
    working_image_w1 = process_image(working_image_w1, True)
    working_image_w2 = process_image(working_image_w2, False)
    return


def process_image(img, is_w1=True):
    if is_w1:
        # set ANYTHING above black to be white
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    else:
        img = cv2.equalizeHist(img)


    # todo tasks from the .docx

    # maybe useful:
    # http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    return img


def mouse_press_callback(event, x, y, flags, param):
    global working_image_w1
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pos:{},{} B:{} G:{} R:{}".format(x, y, working_image_w1[y, x, 0], working_image_w1[y, x, 1],
                                                working_image_w1[y, x, 2]))


main_window_name = "n - cycle images, l - toggle labels, s - save current images, x - exit"
cv2.setMouseCallback(main_window_name, mouse_press_callback)
cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)
image_names = load_image_paths()
load_and_process_next_images()
keep_processing = True
label_images = True


while keep_processing:
    w1, w2 = None, None

    if label_images:
        o_w1_labelled = original_image_w1.copy()
        o_w2_labelled = original_image_w2.copy()
        w_w1_labelled = working_image_w1.copy()
        w_w2_labelled = working_image_w2.copy()
        cv2.putText(o_w1_labelled, 'original w1', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(o_w2_labelled, 'original w2', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(w_w1_labelled, 'processed w1', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(w_w2_labelled, 'processed w2', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        w1 = np.hstack((o_w1_labelled, w_w1_labelled))
        w2 = np.hstack((o_w2_labelled, w_w2_labelled))
    else:
        w1 = np.hstack((original_image_w1, working_image_w1))
        w2 = np.hstack((original_image_w2, working_image_w2))

    both = np.vstack((w1, w2))
    cv2.imshow(main_window_name, both)

    key = cv2.waitKey(40) & 0xFF
    if key == ord('x'):
        keep_processing = False
    elif key == ord('n'):
        load_and_process_next_images()
    elif key == ord('l'):
        label_images = not label_images
    elif key == ord('s'):
        save_images(working_image_w1, working_image_w2)

# todo fix image loop around (when pressing n)
