import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def handle_number(number: int, object_name: str, lower_limit: int) -> None:
    if not isinstance(number, int) or number < lower_limit:
        raise ValueError(
            f'The number of {object_name} must be a positive integer.')


def generate_dataset_lines(
        num_images: int,
        n: int,
        m: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate training images of horizontal or vertical lines on a NxM grid.

    Args:
        num_images (int): Number of images to generate.
        n (int): Number of pixels in the horizontal direction.
        m (int): Number of pixels in the vertical direction.

    Returns:
        Tuple[List[np.ndarray], List[int]]: A tuple containing a list of images
        and a list of labels. Labels are -1 for horizontal lines and 1 for
        vertical lines.
    """
    handle_number(n, 'horizontal pixels', 2)
    handle_number(m, 'vertical pixels', 2)
    handle_number(num_images, 'images', 1)

    logger.debug(f"{num_images} {n}x{m} Images getting created.")

    images = []
    labels = []
    number_of_horizontal_lines = (n - 1) * m
    number_of_vertical_lines = (m - 1) * n
    number_of_pixels = n * m

    hor_array = np.zeros((number_of_horizontal_lines, number_of_pixels))
    ver_array = np.zeros((number_of_vertical_lines, number_of_pixels))

    # Generate horizontal lines
    for row in range(m):
        for col in range(n - 1):
            index = row * n + col
            hor_array[row * (n - 1) + col, [index, index + 1]] = np.pi / 2

    # Generate vertical lines
    for col in range(n):
        for row in range(m - 1):
            index = row * n + col
            ver_array[col * (m - 1) + row, [index, index + n]] = np.pi / 2

    rng = np.random
    for _ in range(num_images):
        if rng.random_integers(0, 2) == 0:  # Horizontal
            labels.append(-1)
            random_image = rng.random_integers(0, number_of_horizontal_lines)
            images.append(hor_array[random_image].copy())
        else:  # Vertical
            labels.append(1)
            random_image = rng.random_integers(0, number_of_vertical_lines)
            images.append(ver_array[random_image].copy())

        # Add noise
        images[-1][images[-1] == 0] = rng.uniform(0, np.pi / 4,
                                                  size=np.sum(images[-1] == 0))

    return images, labels
