from itertools import product, combinations
from PIL import Image, ImageFont, ImageDraw

digits = set(range(1, 10))


def empty(it):
    return next(it, None) is None


def remove_nones(it):
    return (x for x in it if x is not None)


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

    def __repr__(self):
        return 'Square<%d,%d>' % (self.row, self.col)


class Unit:
    def __init__(self, unit_id=0, unit_type=None):
        self.squares = [None] * 9
        self.id = unit_id
        self.type = unit_type

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
        self.units = {'box': [Unit(unit_id, 'box') for unit_id in range(9)],
                      'col': [Unit(unit_id, 'col') for unit_id in range(9)],
                      'row': [Unit(unit_id, 'row') for unit_id in range(9)]}

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

    def units_iter(self):
        for box in self.units['box']:
            yield box
        for row in self.units['row']:
            yield row
        for col in self.units['col']:
            yield col

    def chutes_boxes(self):
        for band in range(3):
            yield 'band', [self.units['box'][band * 3 + col] for col in range(3)]
        for stack in range(3):
            yield 'stack', [self.units['box'][row * 3 + stack] for row in range(3)]

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

    def peer_values(self, square=None, row=None, col=None):
        if square is None:
            square = self.__getitem__((row, col))
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
        self.marks = {(square.row, square.col): set(digits) for square in self.puzzle.missing()}
        self.update()
        self.marks = {(square.row, square.col): digits.difference(self.puzzle.peer_values(square))
                      for square in self.puzzle.missing()}

    def __getitem__(self, square):
        return self.marks[(square.row, square.col)]

    def __contains__(self, square):
        return (square.row, square.col) in self.marks

    def update(self):
        def fix_marks(row, col, marks):
            if self.puzzle[(row, col)].solved():
                return None
            else:
                return marks.difference(self.puzzle.peer_values(row=row, col=col))

        # generator for triplets of row, col and fixed marks for that square
        fixed_marks = ((row, col, fix_marks(row, col, marks)) for ((row, col), marks) in self.marks.items())
        # remove the empty marks
        fixed_marks = ((row, col, marks) for (row, col, marks) in fixed_marks if marks is not None)
        # recreate the dictionary
        as_dict = {(row, col): marks for (row, col, marks) in fixed_marks}
        # keep a set of squares with changed marks, that we will return
        changed_squares = set(k for k in self.marks if k in as_dict and self.marks[k] != as_dict[k])
        # save as_dict as the new marks
        self.marks = as_dict
        return changed_squares

    def single_candidate(self):
        return (self.puzzle[(r, c)] for (r, c) in self.marks if len(self.marks[(r, c)]) == 1)


class AssignmentMove:
    def __init__(self, description, squares_and_digits):
        self.description = description
        self.squares_and_digits = squares_and_digits

    def describe(self):
        return self.description

    def squares(self):
        return [square for square, digit in self.squares_and_digits]

    def execute(self):
        for square, digit in self.squares_and_digits:
            square.digit = digit


class SimplificationMove:
    def __init__(self, pencil_marks, description, squares_and_digits_to_remove):
        self.pencil_marks = pencil_marks
        self.description = description
        self.squares_and_digits_to_remove = squares_and_digits_to_remove

    def describe(self):
        return self.description

    def squares(self):
        return [square for square, to_remove in self.squares_and_digits_to_remove]

    def execute(self):
        for square, digits_to_remove in self.squares_and_digits_to_remove:
            self.pencil_marks[square].difference_update(digits_to_remove)


def show(puzzle, pencil_marks=None, filename=None, display=True, caption=None, highlight=set()):
    margin = 30
    border = 2
    pencil_mark_border = 2
    pencil_mark_square_size = 12
    square_size = pencil_mark_square_size * 3 + pencil_mark_border * 4
    box_size = 3 * square_size + 2 * border
    size = margin * 2 + border * 10 + square_size * 9
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)

    internal_border_color = (160, 160, 160)
    highlight_square_background = (255, 201, 201)
    highlight_square_font = (188, 11, 11)
    pencil_mark_color = (100, 100, 100)

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
            draw.rectangle([(t, l), (t + box_size - 1, l + box_size - 1)], fill=internal_border_color)

        # draw minor grid
        for row, col in product(range(9), range(9)):
            t, l = square_tl(row, col)
            fill = highlight_square_background if (col, row) in highlight else 'white'
            draw.rectangle([(t, l), (t + square_size - 1, l + square_size - 1)], fill=fill)

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
                font_color = highlight_square_font if (row, col) in highlight else 'black'
                draw.text((l, t), str(square.digit), font_color, font=font)
            elif pencil_marks is not None and square in pencil_marks:
                for digit in pencil_marks[square]:
                    draw_pencil_mark(row, col, digit)

    def draw_pencil_mark(row, col, digit):
        t, l = square_tl(row, col)
        pencil_mark_row = (digit - 1) // 3
        pencil_mark_col = (digit - 1) % 3
        l = l + (pencil_mark_col + 1) * pencil_mark_border + pencil_mark_col * pencil_mark_square_size + 3
        t = t + (pencil_mark_row + 1) * pencil_mark_border + pencil_mark_row * pencil_mark_square_size
        font_color = highlight_square_font if (row, col) in highlight else pencil_mark_color
        draw.text((l, t), str(digit), font_color, font=pencil_marks_font)

    draw_grid()
    draw_digits()
    if caption is not None:
        draw.text((5, 5), caption, fill='black', font=given_font)
    if filename is not None:
        img.save(filename)

    if display:
        img.show()


def single_candidate(puzzle):
    def find_digit(square):
        peer_digit_set = set(peer.digit for peer in puzzle.peers(square) if peer.digit is not None)
        return digits.difference(peer_digit_set).pop() if len(peer_digit_set) == 8 else None

    unsolved_boxes = (box for box in puzzle.units['box'] if not box.solved())
    for box in unsolved_boxes:
        moves = ((square, find_digit(square)) for square in box.missing())
        moves = [(square, digit) for (square, digit) in moves if digit is not None]
        if len(moves) > 0:
            return AssignmentMove('Single candidate(s) for box %s' % box.id, moves)


def single_position(puzzle):
    """Find a single position, per unit, where a digit can fit.

    by color: the idea is that you first mentally "color" (or gray out) all the positions that are impossible
    for the digit, and, for each unit, see if there's only one square that isn't colored.
    :param puzzle: Puzzle
    :rtype: AssignmentMove
    """
    for digit in digits:
        squares = []
        colored_out = set(peer for square in puzzle.set_to(digit) for peer in puzzle.peers(square))
        for unit in puzzle.units_iter():
            empty_squares = set(square for square in unit if square.digit is None)
            positions_in_unit = empty_squares.difference(colored_out)
            if len(positions_in_unit) == 1:
                squares.append(positions_in_unit.pop())

            if len(squares) > 0:
                return AssignmentMove('%d has a single position in %s %d' % (digit, unit.type, unit.id + 1),
                                      [(square, digit) for square in squares])


def single_candidate_by_pencil_marks(puzzle, pencil_marks):
    to_assign = list(pencil_marks.single_candidate())
    if len(to_assign) > 0:
        square_and_digit = [(square, list(pencil_marks[square])[0]) for square in to_assign]
        return AssignmentMove('Single pencil mark', square_and_digit)


def single_position_by_pencil_marks(puzzle, pencil_marks):
    for digit in digits:
        squares = []
        for unit in puzzle.units_iter():
            positions_in_unit = [square for square in unit.missing() if digit in pencil_marks[square]]
            if len(positions_in_unit) == 1:
                squares.append(positions_in_unit.pop())
            if len(squares) > 0:
                return AssignmentMove('%d has a single position in %s %d' % (digit, unit.type, unit.id + 1),
                                      [(square, digit) for square in squares])


def n_in_n_simplification(puzzle, pencil_marks, n=2):
    """Implementation of at N in N pencil_mark simplification strategy.

    If there are n squares in a unit, with up to n digit in the union of all their pencil marks,
    then these digits are pinned to these squares, and can't be removed from all other pencil marks
    in the unit.

    For a human solver, the size of n if important - 2 in 2 are easy to spot, 3 in 3 are medium,
     more than that might be tricky.

    :param puzzle: Puzzle
    :param n: int
    :return:
    """
    for unit in puzzle.units_iter():
        # iterate over pairs, triplets etc. of squares in the units
        for positions in combinations(unit.missing(), n):
            # find the set of all candidates for these squares
            candidates = set(digit for square in positions for digit in pencil_marks[square])
            # if the number is exactly n, these candidates can be removed from pencil
            # marks of all other squares in the unit
            if len(candidates) == n:
                others = [square for square in unit.missing() if square not in positions
                          and len(pencil_marks[square].intersection(candidates)) > 0]
                if len(others) > 0:
                    message = '%s are pinned to %d squares in %s %d. They can be removed from %d other squares' \
                              % (str(candidates), n, unit.type, unit.id, len(others))
                    return SimplificationMove(pencil_marks, message, [(square, candidates) for square in others])


class MoveExecutor:
    def __init__(self):
        self.image_counter = 1

    def execute(self, move, puzzle, pencil_marks=None):
        print('executing move %d: %s' % (self.image_counter, move.describe()))
        self.show(move, puzzle, pencil_marks)
        move.execute()
        self.show(move, puzzle, pencil_marks)

        if pencil_marks is not None:
            changed_squares = pencil_marks.update()
            if len(changed_squares) > 0:
                show(puzzle, pencil_marks=pencil_marks, filename='../tmp/%d.png' % self.image_counter, display=False,
                     caption='pencil marks updated', highlight=changed_squares)
                self.image_counter += 1

    def pencil_marks_generated(self, puzzle, pencil_marks):
        show(puzzle, pencil_marks=pencil_marks, filename='../tmp/%d.png' % self.image_counter, display=False,
             caption='pencil marks generated', highlight=set(pencil_marks.marks.keys()))
        self.image_counter += 1

    def show(self, move, puzzle, pencil_marks=None):
        highlight = set((square.row, square.col) for square in move.squares())
        show(puzzle, pencil_marks=pencil_marks, filename='../tmp/%d.png' % self.image_counter, display=False,
             caption=move.describe(), highlight=highlight)
        self.image_counter += 1


def run_assisted_solver(puzzle):
    executor = MoveExecutor()

    # run single_position and single_candidate to exhaustion
    any_move = True
    while any_move:
        any_move = False

        move = single_position(puzzle)
        while move:
            any_move = True
            executor.execute(move, puzzle)
            move = single_position(puzzle)

        move = single_candidate(puzzle)
        while move:
            any_move = True
            executor.execute(move, puzzle)
            move = single_candidate(puzzle)

    # arriving here means that neither single_position nor single_candidate could produce any more
    if puzzle.solved():
        return True

    # create pencil marks
    pencil_marks = PencilMarks(puzzle)
    executor.pencil_marks_generated(puzzle, pencil_marks)

    # because we exhausted single_candidate, single_candidate_by_pencil_marks shouldn't return anything
    assert single_candidate_by_pencil_marks(puzzle, pencil_marks) is None
    assert single_position_by_pencil_marks(puzzle, pencil_marks) is None

    # try 2_in_2 and single_candidate_by_pencil_marks & single_position_by_pencil_marks interchangeably until exhausted
    any_move = True
    while any_move:
        any_move = False

        move = n_in_n_simplification(puzzle, pencil_marks)
        if move:
            any_move = True
            executor.execute(move, puzzle, pencil_marks)

        # exhaust single candidate by pencil mark (extremely visible to humans)!
        move = single_candidate_by_pencil_marks(puzzle, pencil_marks)
        while move:
            any_move = True
            executor.execute(move, puzzle, pencil_marks)
            move = single_candidate_by_pencil_marks(puzzle, pencil_marks)

        move = single_position_by_pencil_marks(puzzle, pencil_marks)
        while move:
            any_move = True
            executor.execute(move, puzzle, pencil_marks)
            # exhaust single candidate by pencil mark (extremely visible to humans)!
            move = single_candidate_by_pencil_marks(puzzle, pencil_marks)
            while move:
                any_move = True
                executor.execute(move, puzzle, pencil_marks)
                move = single_candidate_by_pencil_marks(puzzle, pencil_marks)
            # exhausted, try single position again
            move = single_position_by_pencil_marks(puzzle, pencil_marks)

    return puzzle.solved()
