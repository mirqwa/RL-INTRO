class GridWorld:
    def __init__(self, horizontal, vertical) -> None:
        self.max_horizontal = horizontal
        self.max_vertical = vertical
        self.x = 0
        self.y = 0
        self.done = False
