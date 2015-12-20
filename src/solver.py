def empty(iter):
    return next(iter, None) is None


def find_box(row, col):
    """Return the box number and square number in box"""
    return (row // 3) * 3 + col // 3, row % 3 * 3 + col % 3


def find_row_col(puzzle_position):
    return puzzle_position // 9, puzzle_position % 9


class Square:
    def __init__(self, row, col, digit=None):
        self.row = row
        self.col = col
        self.digit = digit
        self.initial = digit is not None
        self.options = None

    def __eq__(self, other):
        return self.digit == other

    def solved(self):
        return self.digit is not None

    def __str__(self):
        return str(self.digit) if self.digit is not None else '.'


class Unit:
    def __init__(self):
        self.squares = [None] * 9

    def __getitem__(self, key):
        return self.squares[key]

    def __setitem__(self, key, square):
        """
        :type key: int
        :param square: Square
        """
        self.squares[key] = square

    def __contains__(self, digit):
        return empty(square for square in self.squares if square == digit)

    def solved(self):
        return empty(square for square in self.squares if not square.solved())

    def missing(self):
        return [square for square in self.squares if not square.solved()]

    def __str__(self):
        return ''.join(str(square) for square in self.squares)


class Puzzle:
    def __init__(self):
        self.units = {'box': [Unit() for _ in range(9)],
                      'col': [Unit() for _ in range(9)],
                      'row': [Unit() for _ in range(9)]}

    def __getitem__(self, key):
        return self.units['row'][key[0]][key[1]]

    def __setitem__(self, key, value):
        row = key[0]
        col = key[1]
        box_number, box_position = find_box(row, col)
        self.units['row'][row][col] = value
        self.units['col'][col][row] = value
        self.units['box'][box_number][box_position] = value

    def ready(self):
        available = set(square is not None for row in self.units['row'] for square in row)
        return False not in available

    def solved(self):
        return empty(unit for unit in self.units['boxes'] if not unit.solved())

    def unit(self, unit_type, unit_number):
        return self.units[unit_type][unit_number]

    def boxes(self):
        return self.units['box']

    @classmethod
    def from_string(cls, s):
        puzzle = cls()
        # magick! remove everything that isn't a digit ot a . and convert to integers
        characters = [int(c) if c.isdigit() else None for c in s if c.isdigit() or c == '.']
        assert len(characters) == 81
        for i, v in enumerate(characters):
            row, col = find_row_col(i)
            puzzle[(row, col)] = Square(row, col, v)
        return puzzle

    def __str__(self):
        return ''.join(str(row) for row in self.units['row'])




# Iterate over digits. For each number, iterate over boxes missing the
# If the box doesn't contain the number, iterate over square and see how many of them could take
# the number based on col\row sets
def single_position_box(puzzle):
    """Check if the rows and columns pin a single position on boxes. Returns number of iterations to convergence.

    :type puzzle: Puzzle
    :rtype: int
    """

    def single_position_box_digit(box, digit):
        """Returns true if a square was assigned"""
        squares = [square for square in box.missing() if digit not in puzzle.unit('row', square.row)
                   and digit not in puzzle.unit('col', square.col)]
        if len(squares) == 1:
            squares[0].digit = digit
            return True
        return False

    def digit_iteration(digit):
        """Returns true if at least one square was assigned"""
        digit_missing = [box for box in puzzle.boxes() if digit not in box]
        result = [single_position_box_digit(box, digit) for box in digit_missing]
        return True in result

    def iteration():
        """Returns true if at least one square was assigned"""
        result = [digit_iteration(digit) for digit in range(1, 10)]
        return True in result

    counter = 0
    while True:
        if not iteration():
            return counter
        counter += 1