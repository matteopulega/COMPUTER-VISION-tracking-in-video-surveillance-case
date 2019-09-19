import random


class ColorGenerator(object):

    def __init__(self):
        self._used_colors = []

    def generate_color(self):
        while True:
            new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if new_color not in self._used_colors:
                self._used_colors.append(new_color)
                return new_color
