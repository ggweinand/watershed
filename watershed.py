import numpy as np
import numpy.typing as npt

from collections import deque
from typing import List

from utils import Coordinate, State, PriorityQueue

# Type alias for int numpy arrays.
NDArrayInt = npt.NDArray[np.int_]

def watershed(image:NDArrayInt) -> NDArrayInt:
    height, width = limits = image.shape

    state = {}
    label = {}
    queue = deque()
    current_label = 0
    flag = False

    max_level = np.max(image)
    level_coord = [[] for _ in range(max_level + 1)]

    for h in range(height):
        for w in range(width):
            c = Coordinate(h,w)
            level_coord[image[h, w]].append(c)
            state[c] = State.INIT

    for current_level in level_coord:
        # Mask the current level.
        for c in current_level:
            state[c] = State.MASK
            # If a neighbor is a basin or a watershed, add the current pixel to the queue.
            l = [n for n in c.neighbors(limits) if state[n] in [State.WSHED, State.BASIN]]
            if l:
                queue.append(c)
                state[c] = State.INQUEUE

        # Extend existing basins.
        while queue:
            c = queue.popleft()
            for n in c.neighbors(limits):
                match state[n]:
                    case State.BASIN:
                        if state[c] is State.INQUEUE or (state[c] is State.WSHED and flag):
                            label[c] = label[n]
                            state[c] = State.BASIN
                        if state[c] is State.BASIN and label[c] != label[n]:
                            state[c] = State.WSHED
                            flag = False
                    case State.WSHED:
                        if state[c] is State.INQUEUE:
                            state[c] = State.WSHED
                            flag = True
                    case State.MASK:
                        state[n] = State.INQUEUE
                        queue.append(n)

        # Label new basins.
        for c in current_level:
            if state[c] is State.MASK:
                current_label += 1
                state[c] = State.BASIN
                label[c] = current_label
                queue.append(c)

                while queue:
                    p = queue.popleft()
                    for n in p.neighbors(limits):
                        if state[n] is State.MASK:
                            label[n] = current_label
                            state[n] = State.BASIN
                            queue.append(n)

    # Construct segment image.
    result = np.empty_like(image)
    for h in range(height):
        for w in range(width):
            if state[Coordinate(h,w)] is State.WSHED:
                result[h][w] = 0
            else:
                result[h][w] = 255

    return result


def watershed_markers(image:NDArrayInt, markers:NDArrayInt) -> tuple[NDArrayInt, NDArrayInt]:
    height, width = limits = image.shape

    state = {}
    label = {}
    queue = deque()
    flag = False

    max_level = np.max(image)
    level_coord = [[] for _ in range(max_level + 1)]

    for h in range(height):
        for w in range(width):
            c = Coordinate(h,w)
            level_coord[image[h, w]].append(c)
            if markers[h,w]:
                state[c] = State.BASIN
                label[c] = markers[h,w]
            else:
                state[c] = State.INIT
                label[c] = 0

    for current_level in level_coord:
        for c in current_level:
            # If a neighbor is a basin or a watershed, add the current pixel to the queue.
            l = [n for n in c.neighbors(limits) if state[n] in [State.WSHED, State.BASIN]]
            if l:
                queue.append(c)
                state[c] = State.INQUEUE

        # Extend existing basins.
        while queue:
            c = queue.popleft()
            for n in c.neighbors(limits):
                match state[n]:
                    case State.BASIN:
                        if state[c] is State.INQUEUE or (state[c] is State.WSHED and flag):
                            label[c] = label[n]
                            state[c] = State.BASIN
                        if state[c] is State.BASIN and label[c] != label[n]:
                            state[c] = State.WSHED
                            flag = False
                    case State.WSHED:
                        if state[c] is State.INQUEUE:
                            state[c] = State.WSHED
                            flag = True
                    case State.MASK:
                        state[n] = State.INQUEUE
                        queue.append(n)

    # Construct segment image.
    region = np.empty_like(image)
    wshed = np.empty_like(image)
    for h in range(height):
        for w in range(width):
            if state[Coordinate(h,w)] is State.WSHED:
                wshed[h][w] = 0
            else:
                wshed[h][w] = 255
            region[h][w] = label[Coordinate(h,w)]

    return region, wshed


def watershed_markers_cv(image:NDArrayInt, marker:NDArrayInt) -> tuple[NDArrayInt, NDArrayInt]:
    height, width = limits = image.shape

    state: dict[Coordinate, State] = {}
    label: dict[Coordinate, int] = {}
    p_queue = PriorityQueue()
    markers: List[Coordinate] = []
    flag = False

    for h in range(height):
        for w in range(width):
            c = Coordinate(h,w)
            if marker[h,w]:
                state[c] = State.BASIN
                label[c] = marker[h,w]
                markers.append(c)
            else:
                state[c] = State.INIT

    for c in markers:
        for n in c.neighbors(limits):
            if state[n] is State.INIT:
                p_queue.push(n, image[n.h, n.w])
                state[n] = State.INQUEUE

    while not p_queue.empty():
        c = p_queue.pop()

        for n in c.neighbors(limits):
            match state[n]:
                case State.BASIN:
                    if state[c] is State.INQUEUE or (state[c] is State.WSHED and flag):
                        label[c] = label[n]
                        state[c] = State.BASIN
                    if state[c] is State.BASIN and label[c] != label[n]:
                        label[c] = 0
                        state[c] = State.WSHED
                        flag = False
                case State.WSHED:
                    if state[c] is State.INQUEUE:
                        label[c] = 0
                        state[c] = State.WSHED
                        flag = True

        if state[c] is State.BASIN:
            for n in c.neighbors(limits):
                if state[n] is State.INIT:
                    state[n] = State.INQUEUE
                    p_queue.push(n, image[n.h, n.w])

    # Construct segment image.
    region = np.empty_like(image)
    wshed = np.empty_like(image)
    for h in range(height):
        for w in range(width):
            if state[Coordinate(h,w)] is State.WSHED:
                wshed[h,w] = 0
            else:
                wshed[h,w] = 255
            region[h,w] = label[Coordinate(h,w)]

    return region, wshed
