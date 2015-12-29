from itertools import product
from PIL import Image, ImageFont, ImageDraw

digits = set(range(1, 10))


def empty(it):
    return next(it, None) is None


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
        self.given = digit is not None

    @property
    def box(self):
        return find_box(self.row, self.col)[0]

    def solved(self):
        return self.digit is not None

    def __str__(self):
        return str(self.digit) if self.digit is not None else '.'


class Unit:
    def __init__(self, id=0):
        self.squares = [None] * 9
        self.id = id

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
        self.units = {'box': [Unit(id) for id in range(9)],
                      'col': [Unit(id) for id in range(9)],
                      'row': [Unit(id) for id in range(9)]}

    def __getitem__(self, key):
        """
        :type key: (int, int)
        :rtype: Square
        """
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
        """Returns generator of all empty squares.

        :rtype: collections.Iterable[Square]
        """
        return (square for unit in self.units['row'] for square in unit.missing())

    def peers(self, square):
        """Returns all the peers of the input square.

        :type square: Square
        :return: collections.Iterable[Square]
        """
        row = square.row
        col = square.col
        box, _ = find_box(row, col)
        raw = self.units['row'][row].squares + self.units['col'][col].squares + self.units['box'][box].squares
        return (s for s in raw if s != square)

    def peer_values(self, square):
        return set(peer.digit for peer in self.peers(square) if peer.digit is not None)

    def set_to(self, digit):
        return (square for row in self.units['row'] for square in row if square.digit == digit)

    def is_consistent(self):
        squares = (square for row in self.units['row'] for square in row)
        consistent = (s.digit not in self.peer_values(s) for s in squares if s.digit is not None)
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

    @classmethod
    def from_matrix(cls, m):
        def str_digit(d):
            return str(d) if d != 0 else '.'

        lines = (''.join(str_digit(d)) for line in m for d in line)
        return cls.from_string('\n'.join(lines))

    def __str__(self):
        return ''.join(str(row) for row in self.units['row'])

    def pretty(self):
        def do_row(row):
            return '%s %s %s | %s %s %s | %s %s %s\n' % tuple(str(square) for square in self.units['row'][row])

        separator = '------+-------+--------\n'
        return separator.join(['%s %s %s' % (do_row(0), do_row(1), do_row(2)),
                               '%s %s %s' % (do_row(3), do_row(4), do_row(5)),
                               '%s %s %s' % (do_row(6), do_row(7), do_row(8))])


class PencilMarks:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.marks = {}
        self.update()

    def __getitem__(self, square):
        return self.marks[(square.row, square.col)]

    def __contains__(self, square):
        return (square.row, square.col) in self.marks

    def update(self):
        self.marks = {(square.row, square.col): digits.difference(self.puzzle.peer_values(square))
                      for square in self.puzzle.missing()}

    def single_candidate(self):
        return (self.puzzle[(r, c)] for (r, c) in self.marks if len(self.marks[(r, c)]) == 1)


def set_digit(puzzle, square, digit):
    print('setting digit %d at location (%d, %d)' % (digit, square.row, square.col))
    square.digit = digit
    if not puzzle.is_consistent():
        show(puzzle)
        raise ValueError("Puzzle isn't consistent!")


def show(puzzle, pencil_marks=None, filename=None, display=True, caption=None):
    margin = 30
    border = 2
    pencil_mark_border = 2
    pencil_mark_square_size = 12
    square_size = pencil_mark_square_size * 3 + pencil_mark_border * 4
    box_size = 3 * square_size + 2 * border
    size = margin * 2 + border * 10 + square_size * 9
    gray = (160, 160, 160)
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)

    pencil_marks_font = ImageFont.truetype("../fonts/SourceCodePro-Medium.ttf", 10)
    solved_font = ImageFont.truetype("../fonts/SourceCodePro-Medium.ttf", 24)
    given_font = ImageFont.truetype("../fonts/SourceCodePro-Bold.ttf", 24)

    def square_tl(row, col):
        l = margin + (col + 1) * border + col * square_size
        t = margin + (row + 1) * border + row * square_size
        return t, l

    def draw_grid():
        # draw major grid
        draw.rectangle([(margin, margin), (size - margin, size - margin)], outline='black', fill='black')
        for i, j in product(range(3), range(3)):
            t = margin + (i + 1) * border + i * box_size
            l = margin + (j + 1) * border + j * box_size
            draw.rectangle([(t, l), (t + box_size - 1, l + box_size - 1)], fill=gray)

        # draw minor grid
        for row, col in product(range(9), range(9)):
            t, l = square_tl(row, col)
            draw.rectangle([(t, l), (t + square_size - 1, l + square_size - 1)], fill='white')

    def draw_digits():
        for row, col in product(range(9), range(9)):
            square = puzzle[(row, col)]
            if square.solved():
                font = given_font if square.given else solved_font
                w, h = 15, 12  # these are measurement from the results, font.getsize() not helpful
                t_offset, l_offset = 10, 2
                t, l = square_tl(row, col)
                t = t + square_size // 2 - h // 2 - t_offset
                l = l + square_size // 2 - w // 2 - l_offset
                draw.text((l, t), str(square.digit), (0, 0, 0), font=font)
            elif pencil_marks is not None and square in pencil_marks:
                for digit in pencil_marks[square]:
                    draw_pencil_mark(row, col, digit)

    def draw_pencil_mark(row, col, digit):
        t, l = square_tl(row, col)
        pencil_mark_row = (digit - 1) // 3
        pencil_mark_col = (digit - 1) % 3
        l = l + (pencil_mark_col + 1) * pencil_mark_border + pencil_mark_col * pencil_mark_square_size + 3
        t = t + (pencil_mark_row + 1) * pencil_mark_border + pencil_mark_row * pencil_mark_square_size
        draw.text((l, t), str(digit), (100, 100, 100), font=pencil_marks_font)

    draw_grid()
    draw_digits()
    if caption is not None:
        draw.text((5, 5), caption, fill='black', font=given_font)
    if filename is not None:
        img.save(filename)

    if display:
        img.show()


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
            set_digit(puzzle, squares[0], digit)
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
                set_digit(puzzle, square, superset.difference(peer_digit_set).pop())
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
                set_digit(puzzle, square, digit)
                return 1
            return 0

        colored_out = set(peer for square in puzzle.set_to(digit) for peer in puzzle.peers(square))
        units_in_need = [unit for unit in puzzle.units[unit_type] if digit not in unit]
        return sum(find_only_position(unit) for unit in units_in_need)

    def iteration():
        return sum(iteration_for_digit(digit, unit_type) for unit_type, digit in
                   product(('row', 'col', 'box'), range(1, 10)))

    return iteration_runner(iteration)


def single_candidate_by_pencil_marks(puzzle, pencil_marks):
    """Iteratively uses pencil marks to identify cells that have one candidate

    :param puzzle: Puzzle
    :param pencil_marks: PencilMarks
    :rtype: [int]
    """

    def iteration():
        to_assign = list(pencil_marks.single_candidate())
        for i, square in enumerate(to_assign):
            set_digit(puzzle, square, pencil_marks[square].pop())
        pencil_marks.update()
        return len(to_assign)

    return iteration_runner(iteration)


def candidate_line_simplification_iteration(puzzle, pencil_marks, box, digit):
    """This method is not normally called directly, it is used by candidate_line_simplification"""
    def remove(unit_type, unit_id):
        # other squares in unit
        l = (square for square in puzzle.units[unit_type][unit_id] if square.box != box.id)
        # remove those that are solved
        l = (square for square in l if not square.solved())
        # remove the ones that don't have the digit in their pencil marks
        l = (square for square in l if digit in pencil_marks[square])
        # remove the digit from the pencil marks
        l = [pencil_marks[square].remove(digit) for square in l]
        #print("Removing %d from %d squares in %s %d" % (digit, len(l), unit_type, unit_id))
        return len(l)

    positions = set(square for square in box.missing() if digit in pencil_marks[square])
    rows = set(square.row for square in positions)
    r_count = remove('row', rows.pop()) if len(rows) == 1 else 0
    cols = set(square.col for square in positions)
    c_count = remove('col', cols.pop()) if len(cols) == 1 else 0
    return r_count + c_count


def candidate_line_simplification(puzzle, pencil_marks):
    return sum(candidate_line_simplification_iteration(puzzle, pencil_marks, box, digit)
               for box in puzzle.units['box'] for digit in digits)
