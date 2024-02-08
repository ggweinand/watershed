from __future__ import annotations

import heapq

from enum import Enum
from typing import List

State = Enum('State', ['INIT', 'MASK', 'INQUEUE', 'WSHED', 'BASIN'])

class Coordinate:
    def __init__(self, h: int, w:int):
        self.h = h
        self.w = w

    def __str__(self) -> str:
        return f"({self.h},{self.w})"

    def __repr__(self) -> str:
        return f"Coordinate({self.h}, {self.w})"

    def __eq__(self, another: Coordinate):
        return (self.h, self.w) == (another.h, another.w)

    def __hash__(self):
        return hash((self.h, self.w))

    def valid(self, limits: tuple[int, int]) -> bool:
        limit_h, limit_w = limits
        valid_h = 0 <= self.h and self.h < limit_h
        valid_w = 0 <= self.w and self.w < limit_w
        return valid_h and valid_w

    def neighbors(self, limits: tuple[int, int]) -> List[Coordinate]:
        move_h = [-1, 0, 1, -1, 1, -1, 0, 1]
        move_w = [-1, -1, -1, 0, 0, 1, 1, 1]
        unfiltered = [Coordinate(self.h + dh, self.w + dw) for dh, dw in zip(move_h, move_w)]
        return [c for c in unfiltered if c.valid(limits)]

class PriorityQueue:
    def __init__(self):
        self.h: List[tuple[int, int, Coordinate]] = []
        self.entry: int = 0

    def push(self, c: Coordinate, value: int):
        heapq.heappush(self.h, (value, self.entry, c))
        self.entry += 1

    def pop(self) -> Coordinate:
        _, _, c = heapq.heappop(self.h)
        return c

    def empty(self) -> bool:
        return not self.h
