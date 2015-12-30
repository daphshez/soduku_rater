import unittest

from solver import *
from itertools import product

easy1 = """
. . .|1 . 6|. . .
. 6 5|. . .|9 4 .
. . 7|. . .|8 . .
-----+-----+-----
. 5 8|. 4 .|1 7 .
. . .|9 . 1|. . .
. 4 9|. 8 .|6 3 .
-----+-----+-----
. . 3|. . .|5 . .
. 8 2|. . .|7 1 .
. . .|3 . 2|. . .
"""

easy1_solution = """
824 196 357
365 728 941
917 435 826

258 643 179
736 951 284
149 287 635

673 814 592
482 569 713
591 372 468
"""

beginner1 = """
.89 2.7 .6.
652 .38 71.
7.. .61 .9.

5.3 8.9 .4.
476 ... 8.9
... 6.4 537

.6. 41. ..2
.4. 79. ...
.1. ..2 45."""

fiendish_7862 = """
        . . .| . . . | . 9 .
        1 . .| . . . | 8 . 2
        . 5 .| 6 9 . | . 1 .
        -----+-------+------
        8 . .| . 6 . | . . .
        3 6 .| . . 9 | 7 . .
        . 7 5| . . . | 1 . .
        -----+-------+------
        . 3 2| 4 . . | . . .
        . . 8| 7 2 . | 3 . .
        . . .| . 3 6 | . 8 .
        """

bbt = """
... | 893 | ...
..2 | ... | ...
7.3 | .62 | .8.
---------------
..1 | 64. | .9.
..9 | .3. | 8..
.4. | .89 | 2..
---------------
.1. | 42. | 6.9
... | ... | 4..
... | 976 | ...
"""

def compact(grid_string):
    return ''.join(c for c in grid_string if c.isdigit() or c == '.')


class TestDataStructures(unittest.TestCase):
    def test_find_box(self):
        self.assertEqual(find_box(0, 0), (0, 0))
        self.assertEqual(find_box(1, 1), (0, 4))
        self.assertEqual(find_box(4, 4), (4, 4))

    def test_find_row_col(self):
        self.assertEqual(find_row_col(29), (3, 2))
        self.assertEqual(find_row_col(38), (4, 2))
        self.assertEqual(find_row_col(80), (8, 8))

    def test_unit_set_get(self):
        square = Square(1, 1)
        unit = Unit()
        unit[2] = square
        self.assertEqual(unit[2], square)

    def test_unit_solved(self):
        unit = Unit()
        for i in range(9):
            unit[i] = Square(0, i)
        self.assertFalse(unit.solved())
        for i in range(8):
            unit[i].digit = i + 1
            self.assertFalse(unit.solved())
        unit[8].digit = 9
        self.assertTrue(unit.solved())

    def test_unit_missing(self):
        unit = Unit()
        for i in range(9):
            unit[i] = Square(0, i, i + 1)
        unit[0].digit = None
        self.assertEqual(unit.missing(), [unit[0]])

    def test_unit_str(self):
        unit = Unit()
        for i in range(9):
            unit[i] = Square(0, i, i + 1)
        self.assertEqual(str(unit), '123456789')

    def test_puzzle_set_get(self):
        puzzle = Puzzle()
        square = Square(col=2, row=3, digit=1)
        puzzle[(3, 2)] = square
        self.assertEqual(square, puzzle[3, 2])

    def test_puzzle_correct_unit(self):
        puzzle = Puzzle()
        square = Square(col=2, row=3, digit=1)
        puzzle[(3, 2)] = square
        self.assertEqual(square, puzzle.unit('row', 3)[2])
        self.assertEqual(square, puzzle.unit('col', 2)[3])
        self.assertEqual(square, puzzle.unit('box', 3)[2])

    def test_fill_puzzle(self):
        puzzle = Puzzle()
        for row, col in product(range(9), range(9)):
            self.assertFalse(puzzle.ready())
            puzzle[(row, col)] = Square(row, col)
        self.assertTrue(puzzle.ready())

    def test_puzzle_from_string(self):
        puzzle = Puzzle.from_string(fiendish_7862)
        self.assertTrue(puzzle.ready())
        self.assertEqual(str(puzzle), compact(fiendish_7862))

    def test_puzzle_is_consistent(self):
        self.assertTrue(Puzzle.from_string(easy1_solution).is_consistent())
        non_consistent = '1' + easy1_solution.strip()[1:]
        self.assertFalse(Puzzle.from_string(non_consistent).is_consistent())

    def test_puzzle_from_matrix(self):
        m = [[1, 2, 3, 4, 5, 6, 7, 8, 9]
            , [9, 0, 0, 0, 0, 0, 0, 0, 0]
            , [8, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        puzzle = Puzzle.from_matrix(m)
        self.assertEqual(str(puzzle), '123456789' + '9' + (8 * '.') + '8' + (8 * '.') + (6 * 9 * '.'))


class TestOneEmptySquare(unittest.TestCase):
    def runner(self, f):
        grid = '.' + easy1_solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        assignments_per_iter = f(puzzle)
        self.assertEqual(assignments_per_iter, [1, 0])
        self.assertTrue(puzzle.solved())
        self.assertEqual(compact(easy1_solution), str(puzzle))

    def test_single_position_box(self):
        self.runner(single_position_box)

    def test_single_candidate(self):
        self.runner(single_candidate)

    def test_single_position_color(self):
        self.runner(single_position_by_color)

    def test_single_candidate_by_pencil_marks(self):
        grid = '.' + easy1_solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        pencil_marks = PencilMarks(puzzle)
        assignments_per_iter = single_candidate_by_pencil_marks(puzzle, pencil_marks)
        self.assertEqual(assignments_per_iter, [1, 0])
        self.assertTrue(puzzle.solved())
        self.assertEqual(compact(easy1_solution), str(puzzle))


class TestPencilMark(unittest.TestCase):
    def test_one_empty_square(self):
        grid = '.' + easy1_solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        marks = PencilMarks(puzzle)
        self.assertEqual(marks[puzzle[(0, 0)]], {8})


def solving_fiendish_7862():
    puzzle = Puzzle.from_string(fiendish_7862)

    print("Running single position by color")
    print(single_position_by_color(puzzle), puzzle.solved())

    pencil_marks = PencilMarks(puzzle)
    print("running now, single candidate by pencil marks")
    print(single_candidate_by_pencil_marks(puzzle, pencil_marks), puzzle.solved())

    print("Running single position by color")
    print(single_position_by_color(puzzle), puzzle.solved())


    print("Running candidate line pencil mark simplification")
    print(candidate_line_simplification(puzzle, pencil_marks))

    print("Running single candidate by pencil marks, again")
    print(single_candidate_by_pencil_marks(puzzle, pencil_marks))

    print('single position by color, again', single_position_by_color(puzzle))
    print('is solved?', puzzle.solved())
    show(puzzle)

if __name__ == '__main__':
    puzzle = Puzzle.from_string(bbt)
    print("Running single position by color")
    print(single_position_by_color(puzzle), puzzle.solved())
    pencil_marks = PencilMarks(puzzle)

    print("running now, single candidate by pencil marks")
    print(single_candidate_by_pencil_marks(puzzle, pencil_marks), puzzle.solved())

    print("Running candidate line pencil mark simplification")
    print(candidate_line_simplification(puzzle, pencil_marks))
    show(puzzle, pencil_marks)

    print("running now, single candidate by pencil marks")
    print(single_candidate_by_pencil_marks(puzzle, pencil_marks), puzzle.solved())
