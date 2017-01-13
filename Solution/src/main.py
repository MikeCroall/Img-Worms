"""
This script was written to work with Python 3.5.2 using OpenCV 3.1.0, matching the DUDE PCs (in the School)

Script is simply run with no command line arguments, and no manual method calls

To run this script, the provided imgs.txt must be in the same directory, and the folders
    point to by the variable relative_image_folder_path (currently BBBC010_v1_images/, two directories above)
    and the variable relative_image_output_folder_path (currently img_out/, two directories above)
    must be placed correctly relative to the script, or the relative paths adjusted to match their locations.
    As supplied, they will be in the correct places.

imgs.txt is the list of filenames that can be found in the folder BBBC010_v1_images/, except with D24 removed.
    This is because D24 only had a w1 version, and not a w2 version. This is also stated at the top of imgs.txt
    within a comment.

The controls for the window that opens to show the images are shown in the title bar, though is there are a few
    of them, I have outlined them below:

        n - Progress to the next fileID (loads the next w1 and w2 images) (loops back to first ID after last ID)
        b - Return to previous fileID (loads the previous w1 and w2 images)
        l - Toggle labels in the bottom left of each image displayed
        s - Save the currently displayed w1 and w2 images into the folder pointed at by
                relative_image_output_folder_path
        x - exit the window, ending the script
"""
import numpy as np
import cv2

original_image_w1, working_image_w1 = None, None
original_image_w2, working_image_w2 = None, None
relative_image_folder_path = "../../BBBC010_v1_images/"
relative_image_output_folder_path = "../../img_out/"
original_file_name = ""

with open("imgs.txt", "r") as image_names_file:
    # ignore comments (lines beginning with # )
    image_names = [x for x in image_names_file.read().split("\n") if not x.lstrip().startswith("#")]

current_index = -1
current_w1_path = ""
current_w2_path = ""

save_when_processed = False
keep_processing = True
label_images = True


def load_images(w1_path, w2_path):
    """
        Take two file paths and load the image files that the paths point at into global variables
        working_image_w1, working_image_w2 for manipulation, also saving a copy of each original
        in global variables original_image_w1, original_image_w2.

        Keyword arguments:
        w1_path -- the file path to the w1 version of a file
        w2_path -- the file path to the w2 version of a file
    """
    global working_image_w1, original_image_w1
    global working_image_w2, original_image_w2

    # -1 for importing as is (16bit), right shift it then make 8bit to extract more information than importing as 8bit
    w1img16 = cv2.imread(w1_path, -1)
    # ignore 4 least significant bits to be able to use more of the more significant bits
    w1img16b = np.right_shift(w1img16, 4)
    # convert to 8bit
    working_image_w1 = np.array(w1img16b, dtype=np.uint8)
    # save copy for comparing original with processed
    original_image_w1 = working_image_w1.copy()

    # -1 for importing as is (16bit), right shift it then make 8bit to extract more information than importing as 8bit
    w2img16 = cv2.imread(w2_path, -1)
    # ignore 4 least significant bits to be able to use more of the more significant bits
    w2img16b = np.right_shift(w2img16, 4)
    # convert to 8bit
    working_image_w2 = np.array(w2img16b, dtype=np.uint8)
    # save copy for comparing original with processed
    original_image_w2 = working_image_w2.copy()
    return


def cycle_images():
    """
        Finds the next two image filenames to be loaded, constructs the relative file path to them,
        and sends these to load_images(...). Also prints current FileID for reference.
    """
    # Not ALL images have a w1 AND a w2 version - D24 for example has no w2, and so is EXCLUDED from imgs.txt
    global current_index, current_w1_path, current_w2_path, original_file_name
    # progress one image, looping if needed
    current_index = (current_index + 1) % len(image_names)
    # save path to this image as w1
    current_w1_path = relative_image_folder_path + image_names[current_index]
    # progress one image, looping if needed
    current_index = (current_index + 1) % len(image_names)
    # save path to this image as w2
    current_w2_path = relative_image_folder_path + image_names[current_index]
    # save original_file_name for fileID isolation later # todo save just fileID?
    original_file_name = str(image_names[current_index])

    print("Loading images marked {}".format(original_file_name[33:36]))
    # actually load the images from the found paths
    load_images(current_w1_path, current_w2_path)
    return


def save_images(w1, w2):
    """
        Take two image objects, and save them to file in the folder pointed at by
        relative_image_output_folder_path.

        Keyword arguments:
        w1 -- image object to be saved to file as fileid_w1.jpg
        w2 -- image object to be saved to file as fileid_w2.jpg
    """
    global original_file_name
    # isolate the fileID, keeping the _ after it
    file_id = original_file_name[-47:-43]
    cv2.imwrite(relative_image_output_folder_path + file_id + "w1.jpg", w1)
    cv2.imwrite(relative_image_output_folder_path + file_id + "w2.jpg", w2)
    print("Saved {0}w1.jpg and {0}w2.jpg".format(file_id))
    pass


def load_and_process_next_images():
    """
        Cycles to the next pair of images, processes them, and handles saving once processed if variable set
    """
    global working_image_w1, working_image_w2
    cycle_images()
    working_image_w1 = process_image(working_image_w1, True)
    working_image_w2 = process_image(working_image_w2, False)
    if save_when_processed:
        save_images(working_image_w1, working_image_w2)
    return


def process_image(img, is_w1=True):
    """
        Takes an image, knowing if w1 or w2, and processes it according to the tasks given

        Keyword arguments:
        img -- image object to process
        is_w1 -- should be True if img is w1 version, False otherwise
    """
    # w2 can cause border problems, setting the outside area to similar to the border colour
    # begins to alleviate this issue
    if not is_w1:
        img[img < 20] = 50

    # create binary image, where worms are white, background is black (w2 keeps border, which should be removed)
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # w2 needs further border removal
    if not is_w1:
        to_fill = img.copy()
        # fill white from origin
        cv2.floodFill(to_fill, None, (0, 0), 255)
        # make non-white all black
        to_fill[to_fill < 255] = 0
        # dilate white area to grow it over the border
        dilated_flood_fill = cv2.dilate(to_fill, np.ones((5, 5), np.uint8), iterations=2)
        # subtract from original image
        img_bin -= dilated_flood_fill
        # remove extra noise
        img_bin = cv2.medianBlur(img_bin, 5)

    # # contouring modifies image, use a copy
    # contourable_img = img_bin.copy()  # todo use or remove this comment cv2.GaussianBlur(img_bin, (5, 5), 0).copy()
    # # find contours, if w2: largest contour is border to remove
    # returned_image, contours, hierarchy = cv2.findContours(contourable_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # if not is_w1:
    #     # create array of tuples (size, contour), and find contour where size is largest
    #     contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    #     biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    #     # create an image to be subtracted from img_bin
    #     border = np.zeros(img_bin.shape, np.uint8)
    #     # draw known border
    #     cv2.drawContours(border, [biggest_contour], -1, 255, 9)
    #     # find rectangle that bounds border
    #     x, y, w, h = cv2.boundingRect(biggest_contour)
    #
    #     # NOTE were the two rectangle methods below actually rounded rectangles (not included in OpenCV),
    #     # the border corners would NOT get left behind. As OpenCV does not have rounded rectangle drawing
    #     # functionality, my border removal is limited.
    #
    #     # draw over border area in white
    #     cv2.rectangle(border, (x + 5, y + 5), (x + w - 5, y + h - 5), 255, -1)
    #     # draw over center area to stop attached worms from being removed (will leave border corners)
    #     cv2.rectangle(border, (x + 10, y + 10), (x + w - 10, y + h - 10), 0, -1)
    #
    #     # todo MAYBE all smaller than a worm, draw also onto border to get subtracted
    #     # smaller_than_worms = [cs[1] for cs in contour_sizes if cs[0] < 1]
    #     # cv2.drawContours(border, smaller_than_worms, -1, 255, 9)
    #
    #     img_bin -= border
    #     # cv2.namedWindow("temp", cv2.WINDOW_AUTOSIZE) # todo remove this and ...im.show("temp"...
    #     # cv2.imshow("temp", border)

    # todo more tasks from the .docx

    # maybe useful:
    # http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    return img_bin


# prepare window to display results next to originals
main_window_name = "n - next, b - previous, l - toggle labels, s - save current images, x - exit"
cv2.namedWindow(main_window_name, cv2.WINDOW_AUTOSIZE)
load_and_process_next_images()

# until x pressed
while keep_processing:
    both_w1, both_w2 = None, None

    if label_images:
        # create copies, to allow for removing of labels (don't modify original and working
        o_w1_labelled = original_image_w1.copy()
        o_w2_labelled = original_image_w2.copy()
        w_w1_labelled = working_image_w1.copy()
        w_w2_labelled = working_image_w2.copy()
        # place labels uniformly in the bottom left of each image
        cv2.putText(o_w1_labelled, 'original w1', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(o_w2_labelled, 'original w2', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(w_w1_labelled, 'processed w1', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(w_w2_labelled, 'processed w2', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        # combine images for same-window viewing
        both_w1 = np.hstack((o_w1_labelled, w_w1_labelled))
        both_w2 = np.hstack((o_w2_labelled, w_w2_labelled))
    else:
        # combine images for same-window viewing
        both_w1 = np.hstack((original_image_w1, working_image_w1))
        both_w2 = np.hstack((original_image_w2, working_image_w2))

    # final combining of images for same-window viewing
    both = np.vstack((both_w1, both_w2))
    # actually show
    cv2.imshow(main_window_name, both)

    # wait for any given commands
    key = cv2.waitKey(40) & 0xFF
    if key == ord('x'):
        # user wants to exit
        keep_processing = False
    elif key == ord('n'):
        # progress through images
        load_and_process_next_images()
    elif key == ord('b'):
        # reverse through images
        current_index -= 4  # back 4, cycle_images() steps forward twice and handles looping for us
        load_and_process_next_images()
    elif key == ord('l'):
        # toggle label option
        label_images = not label_images
    elif key == ord('s'):
        # trigger save procedure
        save_images(working_image_w1, working_image_w2)
