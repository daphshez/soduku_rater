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
        self.assertTrue(Puzzle.from_string(non_consistent).is_consistent())


class TestSinglePositionBox(unittest.TestCase):
    def test_one_empty_square(self):
        grid = '.' + easy1_solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        n_iter = single_position_box(puzzle)
        self.assertTrue(puzzle.solved())
        self.assertEqual([1, 0], n_iter)
        self.assertEqual(compact(easy1_solution), str(puzzle))

    def test_grid1(self):
        puzzle = Puzzle.from_string(easy1)
        assignments = single_position_box(puzzle)
        self.assertEqual([3, 6, 0], assignments)
        self.assertFalse(puzzle.solved())
        expected_state = '...1.63.7.65...941.17...8...58.4.179...9.1...149.8.63...3.1.5...82...713..13.2...'
        self.assertEqual(expected_state, str(puzzle))


class TestSingleNumber(unittest.TestCase):
    def test_one_empty_square(self):
        grid = '.' + easy1_solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        assignments = single_number(puzzle)
        self.assertTrue(puzzle.solved())
        self.assertEqual([1, 0], assignments)
        self.assertEqual(compact(easy1_solution), str(puzzle))

    def test_grid1(self):
        puzzle = Puzzle.from_string(easy1)
        assignments = single_number(puzzle)
        self.assertTrue(puzzle.solved())
        self.assertEqual([5, 5, 4, 4, 8, 9, 4, 6, 4, 4, 0], assignments)


if __name__ == '__main__':
    # single_position_box doesn't make much advance on easy1
    # single_number solves it, but it takes 11 iterations
    # running single_position_box first doesn't reduce the number of iterations

    # fiendish doesn't lend itself much to either single_position_box or single_number
    puzzle = Puzzle.from_string(fiendish_7862)
    print(single_position_box(puzzle))
    print(puzzle.solved())
    print(puzzle.pretty())

    print(single_number(puzzle))
    print(puzzle.solved())
    print(puzzle.pretty())

    print(single_position_box(puzzle))
    print(puzzle.solved())
    print(puzzle.pretty())