import os.path as osp
from math import ceil, sqrt
from typing import List, Tuple

import cv2
import numpy as np


def get_base_fn(fn: str):
    return osp.basename(fn).split(".")[0]


def tile_images(paths: List[str], row_col: Tuple[int, int] = None, max_size: int = 10, image_size: int = 512):
    if paths is None or len(paths) == 0:
        return

    num_images = len(paths)
    if row_col is not None:
        row, col = min(row_col[0], max_size), min(row_col[1], max_size)
    else:
        fitted_size = ceil(sqrt(num_images))
        row, col = min(fitted_size, max_size), min(fitted_size, max_size)

    def read_images(curr_paths: List[str]):
        images = []
        for p in curr_paths:
            images.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        return images

    images = []
    for i in range(row):
        row_canvas = np.zeros((image_size, image_size*col, 3), dtype=np.uint8)
        row_image = np.hstack(read_images(paths[i*col:(i+1)*col]))
        row_canvas[:row_image.shape[0], :row_image.shape[1]] = row_image
        images.append(row_canvas)
    final_image = np.vstack(images)

    return final_image
