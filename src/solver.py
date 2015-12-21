from itertools import product


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
        self.pencilmarks = None

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
        return not empty(square for square in self.squares if square.digit == digit)

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
        return empty(unit for unit in self.units['box'] if not unit.solved())

    def unit(self, unit_type, unit_number):
        return self.units[unit_type][unit_number]

    def missing(self):
        return (square for unit in self.units['row'] for square in unit.missing())

    def peers(self, square):
        row = square.row
        col = square.col
        box, _ = find_box(row, col)
        raw = self.units['row'][row].squares + self.units['col'][col].squares + self.units['box'][box].squares
        return (s for s in raw if s != square)

    def set_to(self, digit):
        return (square for row in self.units['row'] for square in row if square.digit == digit)

    def is_consistent(self):
        def is_consistent_square(s):
            return True

        squares = (square for row in self.units['row'] for square in row)
        consistent = (is_consistent_square(s) for s in squares)
        return False not in consistent

    @classmethod
    def from_string(cls, s):
        puzzle = cls()
        # magick! remove everything that isn't a digit ot a . and convert to integers
        characters = [int(c) if c.isdigit() else None for c in s if c.isdigit() or c == '.']
        assert len(characters) == 81, len(characters)
        for i, v in enumerate(characters):
            row, col = find_row_col(i)
            puzzle[(row, col)] = Square(row, col, v)
        return puzzle

    def __str__(self):
        return ''.join(str(row) for row in self.units['row'])

    def pretty(self):
        def do_row(row):
            return '%s %s %s | %s %s %s | %s %s %s\n' % tuple(str(square) for square in self.units['row'][row])

        separator = '------+-------+--------\n'
        return separator.join(['%s %s %s' % (do_row(0), do_row(1), do_row(2)),
                               '%s %s %s' % (do_row(3), do_row(4), do_row(5)),
                               '%s %s %s' % (do_row(6), do_row(7), do_row(8))])


def iteration_runner(f):
    """Runs the function f indefinitely until it returns 0. Returns the list of value returned by f.

    :type f: function
    :rtype: list[int]
    """
    assignments = []
    while True:
        assignments.append(f())
        if assignments[-1] == 0:
            break
    return assignments


# Iterate over digits. For each number, iterate over boxes missing the
# If the box doesn't contain the number, iterate over square and see how many of them could take
# the number based on col\row sets
def single_position_box(puzzle):
    """Check if the rows and columns pin a single position on boxes. Returns number of iterations to convergence.

    :type puzzle: Puzzle
    :rtype: list[int]
    """

    def single_position_box_digit(box, digit):
        """Returns true if a square was assigned"""
        squares = [square for square in box.missing() if digit not in puzzle.unit('row', square.row)
                   and digit not in puzzle.unit('col', square.col)]
        if len(squares) == 1:
            squares[0].digit = digit
            return 1
        return 0

    def digit_iteration(digit):
        """Returns the number of squares assigned"""
        digit_missing = [box for box in puzzle.units['box'] if digit not in box]
        return sum(single_position_box_digit(box, digit) for box in digit_missing)

    def iteration():
        """Returns the number of squares assigned"""
        return sum(digit_iteration(digit) for digit in range(1, 10))

    return iteration_runner(iteration)


def single_candidate(puzzle):
    """Find squares that are pinned by all their peers to a single number

    :type puzzle: Puzzle
    """
    superset = set(range(1, 10))

    def iteration():
        assignments = 0
        for square in puzzle.missing():
            peer_digit_set = set(peer.digit for peer in puzzle.peers(square) if peer.digit is not None)
            if len(peer_digit_set) == 8:
                square.digit = superset.difference(peer_digit_set).pop()
                assignments += 1
        return assignments

    return iteration_runner(iteration)


def single_position_by_color(puzzle):
    def iteration_for_digit(digit, unit_type):
        def find_only_position(unit):
            # looks for a single non-colored empty square
            # if found one, set the value to digit and return 1, else return 0
            empty_squares = set(square for square in unit if square.digit is None)
            non_colored_empty_squares = empty_squares.difference(colored_out)
            if len(non_colored_empty_squares) == 1:
                square = non_colored_empty_squares.pop()
                square.digit = digit
                return 1
            return 0

        colored_out = set(peer for square in puzzle.set_to(digit) for peer in puzzle.peers(square))
        units_in_need = [unit for unit in puzzle.units[unit_type] if digit not in unit]
        return sum(find_only_position(unit) for unit in units_in_need)

    def iteration():
        return sum(iteration_for_digit(digit, unit_type) for unit_type, digit in
                   product(('row', 'col', 'box'), range(1, 10)))

    return iteration_runner(iteration)


def generate_pencilmarks(puzzle):
    all = set(range(1, 10))
    for square in puzzle.missing():
        peer_digits = set(peer.digit for peer in puzzle.peers(square) if peer.digit is not None)
        square.pencilmarks = all.difference(peer_digits)