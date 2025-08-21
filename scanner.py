from omr_utils import *
import sys
from align import align_sheet

GREEN = (0, 255, 0)
BLUE = (255,0,0)
RED = (0,0, 255)


def scanner(img):
    """
    Perform OMR (Optical Mark Recognition) scanning on an input sheet image.

    Steps:
        1. Convert image to grayscale.
        2. Apply Canny edge detection.
        3. Detect circular bubbles using circularity.
        4. Remove overlapping bubbles in the first row.
        5. Identify number of options and columns.
        6. Compute first and last columns to derive total questions.
        7. Generate and arrange bubble grids.
        8. Threshold grayscale image for filled bubble detection.
        9. For each question grid, check selected option(s).

    Parameters:
        img (ndarray): Input OMR sheet image.

    Returns:
        grid (ndarray): Final grid of bubble coordinates shaped (questions, options, 2).
        total_ques (int): Total number of detected questions.
        key (dict): Detected key mapping {question_index: selected_option}.
        Radius (int): Maximum bubble radius observed across detected bubbles.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection parameters
    t_lower = 50
    t_upper = 150
    aperture_size = 5
    edge = cv2.Canny(gray, t_lower, t_upper, apertureSize=aperture_size)

    # Detect bubbles by circularity
    bubbles, contours, Radius = find_bubbles_from_circularity(
        edge, min_area=150, circ_threshold=0.75
    )

    # Clean overlapping bubbles in the first row
    first_row = remove_overlapping_bubbles_row(bubbles[0])

    # Count options and columns
    options, columns = count_options_and_columns(first_row)
    if not (columns and options):
        return None, None, None, None

    # Find first and last column
    first_col = get_first_column(bubbles)
    last_col = get_last_column(bubbles)
    # Compute total questions based on column distribution
    total_ques = (columns - 1) * len(first_col) + len(last_col)


    # Generate left and right grids

    left_grid, right_grid = generate_bubble_grid(first_row, first_col, last_col, options)

    # Reshape left grid
    left_grid = left_grid.transpose(1, 0, 2)
    rows, cols, _ = left_grid.shape

    blocks = []
    for i in range(0, cols, options):
        block = left_grid[:, i:i + options, :]  # shape (rows, options, 2)
        blocks.append(block)

    # Transpose right grid and append
    right_grid = right_grid.transpose(1, 0, 2)
    blocks.append(right_grid)

    # Combine all blocks into one grid
    grid = np.vstack(blocks)


    # Threshold grayscale for bubble fill detection
    _, im_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Detect answers
    key = {}
    for idx, ques in enumerate(grid):
        res = check_single_question(im_bw, ques, Radius, 0.8)
        key[idx + 1] = res

    return grid, total_ques, key, Radius


if __name__ == "__main__":
    path = sys.argv[1]

    # Load and align the input image
    img = cv2.imread(path)
    img = align_sheet(img)

    # Perform scanning
    grid, total_ques, key, radius = scanner(img)
    if grid is None:
        print("Image is not clear")
        exit(0)


    for idx,value in key.items():
        if value <1:
            color = RED if value == -1 else BLUE
            for j in range(grid.shape[1]):  # cols
                x, y = grid[idx -1, j].astype(int)
                cv2.circle(img, (x, y), radius=radius, color=color, thickness=2)
        else:
            x, y = grid[idx - 1, value-1].astype(int)
            cv2.circle(img, (x, y), radius=radius, color=GREEN, thickness=2)


    # Display annotated image
    show(img)

    # Print summary
    print("total questions detected:", total_ques)
    print("detected key:", key)
