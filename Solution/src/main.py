"""
This script was written to work with Python 3.5.2 using OpenCV 3.1.0, matching the DUDE PCs (in the School)

Script is simply run with no command line arguments, and no manual method calls

To run this script, the provided imgs.txt must be in the same directory, and the folders
    point to by the variable relative_image_folder_path (currently BBBC010_v1_images/, two directories above)
    and the variable relative_image_output_folder_path (currently img_out/, two directories above)
    must be placed correctly relative to the script, or the relative paths adjusted to match their locations.
    As supplied, they will be in the correct places. The same goes for other relative_image_... folder locations.

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
relative_image_ground_truth_folder_path = "../../BBBC010_v1_foreground/"
relative_image_output_folder_path = "../../img_out/"
file_id = ""

with open("imgs.txt", "r") as image_names_file:
    # ignore comments (lines beginning with # )
    image_names = [x for x in image_names_file.read().split("\n") if not x.lstrip().startswith("#")]

current_index = -1
current_w1_path = ""
current_w2_path = ""

save_when_processed = True
save_individual_worms = False
keep_processing = True
label_images = True

auto_advance = False


def load_images(w1_path, w2_path):
    """
    Take two file paths and load the image files that the paths point at into global variables
    working_image_w1, working_image_w2 for manipulation, also saving a copy of each original
    in global variables original_image_w1, original_image_w2.

    :param w1_path: The file path to the w1 version of a file
    :param w2_path: he file path to the w2 version of a file
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

    # convert to colour for displaying with coloured processed images
    original_image_w1 = cv2.cvtColor(original_image_w1, cv2.COLOR_GRAY2BGR)
    original_image_w2 = cv2.cvtColor(original_image_w2, cv2.COLOR_GRAY2BGR)

    return


def cycle_images():
    """
    Finds the next two image filenames to be loaded, constructs the relative file path to them,
    and sends these to load_images(...). Also prints current FileID for reference.
    """
    # Not ALL images have a w1 AND a w2 version - D24 for example has no w2, and so is EXCLUDED from imgs.txt
    global current_index, current_w1_path, current_w2_path, file_id
    # progress one image, looping if needed
    current_index = (current_index + 1) % len(image_names)
    # save path to this image as w1
    current_w1_path = relative_image_folder_path + image_names[current_index]
    # progress one image, looping if needed
    current_index = (current_index + 1) % len(image_names)
    # save path to this image as w2
    current_w2_path = relative_image_folder_path + image_names[current_index]
    # save file_id for saving related files later
    file_id = str(image_names[current_index])[33:36]

    print("\nLoading images marked {}".format(file_id))
    # actually load the images from the found paths
    load_images(current_w1_path, current_w2_path)
    return


def save_images(w1, w2):
    """
    Take two image objects, and save them to file in the folder pointed at by
    relative_image_output_folder_path.

    :param w1: Image object to be saved to file as fileid_w1.jpg
    :param w2: Image object to be saved to file as fileid_w2.jpg
    """
    global file_id
    cv2.imwrite(relative_image_output_folder_path + file_id + "_w1.jpg", w1)
    cv2.imwrite(relative_image_output_folder_path + file_id + "_w2.jpg", w2)
    print("Saved {0}_w1.jpg and {0}_w2.jpg".format(file_id))
    return


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


def step_1_isolate_worms(img, is_w1=True):
    """
    Takes an image, knowing if w1 or w2, and isolates the worms from the background as best as possible

    :param img: Image object to process
    :param is_w1: Should be True if img is w1 version, False otherwise
    :return: Processed Image object with worms as isolated as possible
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

    # fill in empty spots within worms
    morph_close_iterations = 3 if is_w1 else 2
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=morph_close_iterations)

    # remove big blotches that aren't full worms
    # NOTE this can remove parts of worms that weren't processed possibly,
    #       though these can often be found in the w1 version of the same image
    if not is_w1:
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    return img_bin


def calculate_percentage_similarity(img_a, img_b):
    """
    Calculates the percentage of pixels identical between two images

    :param img_a: The first image to compare
    :param img_b: The second image to compare
    :return: The percentage of pixels identical between img_a and img_b
    """
    # ensure both images are still binary to avoid differences by 1 pixel causing ~0% similarity
    img_a_2 = cv2.adaptiveThreshold(img_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    img_b_2 = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # find difference
    difference = img_b_2 - img_a_2

    # count white (actually counting non-black) pixels in difference
    white = cv2.countNonZero(difference)
    # count total pixels in difference
    total = difference.size  # originally was: total = difference.shape[0] * difference.shape[1]
    # calculate percentage black pixels (black means identical between img_a and img_b)
    return 100 * ((total - white) / total)


def step_1b_compare_to_ground_truth(img_1, is_w1):
    """
    Takes an image, loads the corresponding ground truth, finds percentage similarity

    :param img_1: Image to compare against the ground truth image
    :param is_w1: Should be True if img_1 is w1 version, False otherwise
    """
    global file_id, relative_image_ground_truth_folder_path
    # load ground truth from disk
    ground_truth = cv2.imread(relative_image_ground_truth_folder_path + file_id + "_binary.png", -1)
    # convert ground truth to be single channel, to match img_1
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    if ground_truth is not None and ground_truth.shape == img_1.shape:
        percentage_similarity = calculate_percentage_similarity(img_1, ground_truth)
        print("\t{} {} - matches ground similarity {:.2f}%".format(file_id, "w1" if is_w1 else "w2",
                                                                   percentage_similarity))
    else:
        print("\tGround truth image for {} could not be loaded for comparison," +
              "or does not match shape of img_1".format(file_id))

    return


def step_2_watershed(img):
    """
    Takes an image, knowing if w1 or w2, and uses the watershed algorithm to mark the worms

    :param img: Image object to process
    :param is_w1: Should be True if img is w1 version, False otherwise
    :return: Processed Image object with worms marked where possible
    """
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find what is definitely background
    definite_background = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=3)

    # find what is definitely foreground
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    ret, definite_foreground = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # convert foreground and background to 8bit
    definite_foreground = np.uint8(definite_foreground)
    definite_background = np.uint8(definite_background)

    # calculate what is then unknown
    unknown = cv2.subtract(definite_background, definite_foreground)

    ret, markers = cv2.connectedComponents(definite_foreground)

    markers += 1
    markers *= 20
    markers[unknown == 255] = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]

    markers = np.uint8(markers)

    return img, markers


def step_2b_save_individual_worms(watershed_markers, is_w1):
    """
    Takes a marker image from the watershed process, and saves each worm in it's own file

    :param watershed_markers: The marker image from the watershed process
    :param is_w1: Should be True if img is w1 version, False otherwise
    """
    global file_id, relative_image_output_folder_path, save_individual_worms
    border_colour = watershed_markers[0, 0]
    background_colour = watershed_markers[5, 5]
    colours_to_save = []

    for x in range(watershed_markers.shape[0]):
        for y in range(watershed_markers.shape[1]):
            pixel_colour = watershed_markers[x, y]
            if pixel_colour != background_colour and pixel_colour != border_colour:
                if pixel_colour not in colours_to_save:
                    colours_to_save.append(pixel_colour)

    print("\t{} {} - {} worms identified".format(file_id, "w1" if is_w1 else "w2", len(colours_to_save)))

    if save_individual_worms:
        print("\t{} {} - Saving individual worms in img_out/separated/".format(file_id, "w1" if is_w1 else "w2"))
        counter = 1
        for colour in colours_to_save:
            img_to_save = watershed_markers.copy()
            img_to_save[watershed_markers == colour] = 255
            img_to_save[watershed_markers != colour] = 0
            cv2.imwrite(
                relative_image_output_folder_path + "separated/{}_{}_{}.jpg".format(file_id, "w1" if is_w1 else "w2",
                                                                                    str(counter)), img_to_save)
            counter += 1
    return


def step_3_classify_dead_or_alive(img):
    """
    Takes an image, colours in alive worms green, and dead worms red, returns coloured in image

    :param img: Image object to identify and colour worms in
    :return: Image object with worms coloured in depending on whether they are dead or alive
    """
    # contouring modifies image, use a copy
    contourable_img = np.uint8(img.copy())
    # convert coloured image to grey scale
    contourable_img = cv2.cvtColor(contourable_img, cv2.COLOR_BGR2GRAY)
    # find contours, if w2: largest contour is border to remove
    returned_image, contours, hierarchy = cv2.findContours(contourable_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # assess each contour individually
    for contour in contours:
        # find smallest bounding rectangle of contour
        rotated_rectangle = cv2.minAreaRect(contour)
        # extract width and height from rotated rectangle
        width, height = rotated_rectangle[1]
        # ignore tiny pixel clusters
        if width > 2 and height > 2:
            # find ratio, ensuring >= 1
            width_height_ratio = max(width, height) / min(width, height)
            if width_height_ratio < 2.5:
                # alive, fill green
                cv2.drawContours(img, [contour], 0, (0, 255, 0), -1)
            else:
                # dead, fill red
                cv2.drawContours(img, [contour], 0, (0, 0, 255), -1)

            # draw bounding rotated rectangle around coloured in worm, in purple
            box = cv2.boxPoints(rotated_rectangle)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 0, 255), 2)

    return img


def process_image(img, is_w1=True):
    """
    Takes an image, knowing if w1 or w2, and processes it according to the tasks given

    :param img: Image object to process
    :param is_w1: Should be True if img is w1 version, False otherwise
    :return: Fully processed Image object
    """
    # Isolate the worms from the background/border
    img_1 = step_1_isolate_worms(img, is_w1)

    # Calculate percentage similarity to provided ground truth
    step_1b_compare_to_ground_truth(img_1, is_w1)

    # Isolate individual worms in separate colours (watershed_markers is grey scale,
    #                                               img_2 is coloured with worms outlined)
    img_2, watershed_markers = step_2_watershed(img_1)

    # Use watershed_markers to save individual worms
    step_2b_save_individual_worms(watershed_markers, is_w1)

    # Find contours and determine shape for dead/alive classification (worms boxed with purple,
    #                                                               alive are coloured green, dead are coloured red)
    img_3 = step_3_classify_dead_or_alive(img_2)

    return img_3


# prepare window to display results next to originals
main_window_name = "n - next, b - previous, l - toggle labels, s - save current images, x - exit"
cv2.namedWindow(main_window_name, cv2.WINDOW_AUTOSIZE)
load_and_process_next_images()

# todo print info about key commands, save folders, load folders, etc etc

# todo go through all code and comment/docstring appropriately

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
    elif auto_advance:
        # automatically load and process next images
        load_and_process_next_images()
