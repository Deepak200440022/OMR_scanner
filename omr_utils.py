import cv2
import math
import numpy as np
from collections import Counter


def circularity(cnt, min_area, tol=2.0):
    """
    Compute the circularity of a contour.

    Parameters:
        cnt (ndarray): Contour points.
        min_area (float): Minimum contour area for validity.
        tol (float): Area tolerance factor.

    Returns:
        float: Circularity value (0 if area too small).
    """
    area = cv2.contourArea(cnt)
    if area < min_area * (1 - tol):
        return 0
    param = cv2.arcLength(cnt, True)
    return (4 * math.pi * area) / pow(param, 2)


def show(img, title="sample"):
    """
    Display an image in a window and wait for user keypress.

    Parameters:
        img (ndarray): Image to display.
        title (str): Window title.
    """
    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(0) & 0xFF  # get lower 8 bits
        if key == ord('q'):  # ord('q') = 113
            break
    cv2.destroyAllWindows()


def find_bubbles_from_circularity(
    filled_img,
    min_area=50,
    circ_threshold=0.7,
    tol=0.0,
    row_tol=15
):
    """
    Detect bubble-like contours in a binary image using circularity.

    Parameters:
        filled_img (ndarray): Binary image (0/255) with filled contours.
        min_area (int): Minimum area for valid contour.
        circ_threshold (float): Minimum circularity to consider as bubble.
        tol (float): Area tolerance factor.
        row_tol (int): Y-axis tolerance for grouping bubbles into rows.

    Returns:
        tuple:
            bubble_matrix (list[list]): Rows of bubbles [(x, y, r), ...].
            contour_matrix (list[list]): Same as bubble_matrix but with contours.
            most_common_radius (int or None): Most frequent bubble radius.
    """
    contours, _ = cv2.findContours(filled_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bubbles = []
    valid_contours = []
    radii = []

    # Filter contours by circularity
    for cnt in contours:
        circ = circularity(cnt, min_area, tol)
        if circ >= circ_threshold:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            r = int(r)
            bubbles.append((int(x), int(y), r))
            valid_contours.append(cnt)
            radii.append(r)

    # Sort bubbles by Y then X
    sorted_data = sorted(zip(bubbles, valid_contours), key=lambda d: (d[0][1], d[0][0]))

    bubble_matrix, contour_matrix = [], []
    current_row_bubbles, current_row_contours = [], []
    last_y = None

    # Group into rows
    for (bubble, cnt) in sorted_data:
        x, y, r = bubble
        if last_y is None or abs(y - last_y) <= row_tol:
            current_row_bubbles.append(bubble)
            current_row_contours.append(cnt)
            last_y = y if last_y is None else (last_y + y) // 2
        else:
            bubble_matrix.append(sorted(current_row_bubbles, key=lambda b: b[0]))
            contour_matrix.append(sorted(current_row_contours, key=lambda c: cv2.boundingRect(c)[0]))
            current_row_bubbles, current_row_contours = [bubble], [cnt]
            last_y = y

    if current_row_bubbles:
        bubble_matrix.append(sorted(current_row_bubbles, key=lambda b: b[0]))
        contour_matrix.append(sorted(current_row_contours, key=lambda c: cv2.boundingRect(c)[0]))

    most_common_radius = Counter(radii).most_common(1)[0][0] if radii else None
    return bubble_matrix, contour_matrix, most_common_radius


def remove_overlapping_bubbles_row(row):
    """
    Remove overlapping bubbles in a single row.

    Parameters:
        row (list): List of (x, y, r) sorted left-to-right.

    Returns:
        ndarray: Cleaned row of bubbles.
    """
    if len(row) == 0:
        return np.empty((0, 3), dtype=int)

    row = np.array(row, dtype=float)
    keep = [row[0]]

    # Compare each bubble with the last kept bubble
    for i in range(1, len(row)):
        x1, y1, r1 = keep[-1]
        x2, y2, r2 = row[i]
        dist = np.hypot(x2 - x1, y2 - y1)

        if dist < min(r1, r2):
            if r2 > r1:  # Replace smaller with larger
                keep[-1] = row[i]
        else:
            keep.append(row[i])

    return np.array(keep, dtype=int)


def get_first_column(bubbles_matrix, x_tolerance=20):
    """
    Extract and clean the first column of bubbles.

    Parameters:
        bubbles_matrix (list): Rows of bubbles [(x, y, r), ...].
        x_tolerance (int): Allowed tolerance in X.

    Returns:
        ndarray: First column bubbles.
    """
    if not bubbles_matrix:
        return []

    min_x = min(row[0][0] for row in bubbles_matrix if row)
    first_col = [row[0] for row in bubbles_matrix if row and abs(row[0][0] - min_x) <= x_tolerance]
    return remove_overlapping_bubbles_row(first_col)


def get_last_column(bubbles_matrix, x_tolerance=20):
    """
    Extract and clean the last column of bubbles.

    Parameters:
        bubbles_matrix (list): Rows of bubbles [(x, y, r), ...].
        x_tolerance (int): Allowed tolerance in X.

    Returns:
        ndarray: Last column bubbles.
    """
    if not bubbles_matrix:
        return np.array([])

    max_x = max(row[-1][0] for row in bubbles_matrix if row)
    last_col = [row[-1] for row in bubbles_matrix if row and abs(row[-1][0] - max_x) <= x_tolerance]
    return remove_overlapping_bubbles_row(last_col)


def count_options_and_columns(first_row):
    """
    Estimate options per question and total columns.

    Parameters:
        first_row (ndarray): First row of bubbles (x, y, r).

    Returns:
        tuple: (options_per_question, total_columns).
    """
    if len(first_row) < 3:
        return 0, 0

    gaps = [first_row[i + 1][0] - first_row[i][0] for i in range(2)]
    base_gap = np.median(gaps)

    options_per_question = len(first_row)
    for i in range(len(first_row) - 1):
        if (first_row[i + 1][0] - first_row[i][0]) > base_gap * 2:
            options_per_question = i + 1
            break

    total_columns = len(first_row) // options_per_question
    return options_per_question, total_columns


def find_longest_horizontal_line(
    img,
    canny_img,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    min_line_length=100,
    max_line_gap=10,
    horizontal_tolerance_deg=5.0,
    margin=20,
    perpendicular_tolerance_deg=5.0,
):
    """
    Find the longest horizontal line in an image using Hough transform.

    Parameters:
        img (ndarray): Original image.
        canny_img (ndarray): Canny edge image.
        rho, theta, threshold, min_line_length, max_line_gap: Hough transform params.
        horizontal_tolerance_deg (float): Angle tolerance for horizontal.
        margin (int): Margin exclusion.
        perpendicular_tolerance_deg (float): Angle tolerance for perpendicular.

    Returns:
        tuple: (best_line (x1,y1,x2,y2), best_angle)
    """
    lines = cv2.HoughLinesP(
        canny_img, rho=rho, theta=theta,
        threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    if lines is None:
        return None, None

    h, w = canny_img.shape[:2]
    candidates = []

    # Filter horizontal candidates
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if (x1 < margin or x2 < margin or y1 < margin or y2 < margin or
            x1 > w - margin or x2 > w - margin or y1 > h - margin or y2 > h - margin):
            continue

        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        if angle > 90: angle -= 180
        elif angle < -90: angle += 180

        if abs(angle) <= horizontal_tolerance_deg:
            candidates.append(((int(x1), int(y1), int(x2), int(y2)), angle, np.hypot(dx, dy)))

    if not candidates:
        return None, None

    best_with_perp, best_len = None, 0.0
    # Check for perpendicular pairing
    for line, angle, length in candidates:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx2, dy2 = x2 - x1, y2 - y1
            angle2 = np.degrees(np.arctan2(dy2, dx2))
            if angle2 > 90: angle2 -= 180
            elif angle2 < -90: angle2 += 180

            if abs(abs(angle - angle2) - 90) <= perpendicular_tolerance_deg:
                if length > best_len:
                    best_len = length
                    best_with_perp = (line, angle)

    if best_with_perp:
        return best_with_perp
    return max(candidates, key=lambda x: x[2])[:2]


def generate_bubble_grid(first_row, first_col, last_col, options):
    """
    Generate bubble grid coordinates from anchor rows and columns.

    Parameters:
        first_row (ndarray): Top row of bubbles.
        first_col (ndarray): First column bubbles.
        last_col (ndarray): Last column bubbles.
        options (int): Options per question.

    Returns:
        tuple: (left_grid, right_grid) as ndarray.
    """
    n_rows = len(first_col)
    n_cols = len(first_row)
    left_cols = n_cols - options

    # Left grid
    left_grid = [[None for _ in range(n_rows)] for _ in range(left_cols)]
    for j in range(left_cols):
        left_grid[j][0] = (first_row[j][0], first_row[j][1])
    for i in range(n_rows):
        left_grid[0][i] = (first_col[i][0], first_col[i][1])

    for i in range(1, n_rows):
        for j in range(1, left_cols):
            x = left_grid[j - 1][i][0] + left_grid[j][i - 1][0] - left_grid[j - 1][i - 1][0]
            y = left_grid[j - 1][i][1] + left_grid[j][i - 1][1] - left_grid[j - 1][i - 1][1]
            left_grid[j][i] = (x, y)

    # Right grid
    n_rows_right = len(last_col)
    n_cols_right = options
    right_grid = [[None for _ in range(n_rows_right)] for _ in range(n_cols_right)]

    for j in range(n_cols_right):
        right_grid[j][0] = (first_row[-n_cols_right + j][0], first_row[-n_cols_right + j][1])
    for i in range(n_rows_right):
        right_grid[-1][i] = (last_col[i][0], last_col[i][1])

    for i in range(1, n_rows_right):
        for j in range(n_cols_right - 2, -1, -1):
            x = right_grid[j + 1][i][0] + right_grid[j][i - 1][0] - right_grid[j + 1][i - 1][0]
            y = right_grid[j + 1][i][1] + right_grid[j][i - 1][1] - right_grid[j + 1][i - 1][1]
            right_grid[j][i] = (x, y)

    return np.array(left_grid), np.array(right_grid)


def check_single_question(binary_img, coords, radius, threshold):
    """
    Check which option(s) are filled for a single question.

    Parameters:
        binary_img (ndarray): Binary answer sheet image.
        coords (list): Bubble coordinates [(x, y), ...].
        radius (int): Bubble radius.
        threshold (float): Fill ratio threshold.

    Returns:
        int: 0 if none filled, option index+1 if one filled, -1 if multiple filled.
    """
    marked_indices = []

    for idx, (cx, cy) in enumerate(coords):
        roi = binary_img[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        filled_pixels = np.sum(roi == 0)
        total_pixels = np.sum(roi == 255)

        fill_ratio = filled_pixels / total_pixels
        if fill_ratio >= threshold:
            marked_indices.append(idx)

    if len(marked_indices) == 0:
        return 0
    elif len(marked_indices) == 1:
        return marked_indices[0] + 1
    return -1

