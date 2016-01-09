import unittest

from solver import *
from itertools import product
from collections import namedtuple

Example = namedtuple('Example', ('source', 'level', 'description', 'puzzle', 'solution'))

examples = {}

examples['easy'] = Example('The Times', 'easy', 'Published December 2015', """
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
""", """
    824 196 357
    365 728 941
    917 435 826

    258 643 179
    736 951 284
    149 287 635

    673 814 592
    482 569 713
    591 372 468
""")

examples['begginer'] = Example('App', 'beginner', '', """
    .89 2.7 .6.
    652 .38 71.
    7.. .61 .9.
    -----------
    5.3 8.9 .4.
    476 ... 8.9
    ... 6.4 537

    .6. 41. ..2
    .4. 79. ...
    .1. ..2 45.""", None)

examples['fiendish'] = Example('The Times', 'fiendish', 'Published December 2015', """
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
        """, None)

examples['bbt'] = Example('App', 'master', '', """
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
        """, None)

examples['bbt2'] = Example('App', 'master', 'bbt2', """
        . 1 4 | 5 . . | . 7 .
        8 . . | . . 4 | 6 . 1
        . . . | . . . | . 2 .
        ---------------------
        . . 6 | 4 . . | 1 . .
        . . . | . 3 . | . . .
        . . 1 | . . 9 | 2 . .
        ---------------------
        . 2 . | . . . | . . .
        3 . 5 | 6 . . | . . 2
        . 8 . | . . 7 | 5 6 .
        """, None)

examples['horrible'] = Example('App', 'master', 'cannot be solved with only 2-in-2 and 3-in-3 simplification', """
        1 5 . | . 7 3 | . . .
        . . . | 9 . . | . . 7
        9 . . | . . . | . . .
        ---------------------
        5 . 9 | 1 . . | 7 . .
        3 . 8 | 5 4 7 | 9 . 1
        . . 1 | . . 9 | 4 . 3
        ---------------------
        . . . | . . . | . . 8
        2 . . | . . 5 | . . .
        . . . | 2 9 . | . 3 4
        """, None)

examples['bbt3'] = Example('App', 'Master', '?', """
. . . | 1 9 5 | . . .
. . . | . . . | . . 6
. 3 . |  . 4 6 | 5 . 1
----------------------
. 6 . | 5 7 . | . . 4
5 . . | . 8 . | . . 7
3 . . | . 6 1 | . 5 .
---------------------
8 . 9 | 4 1 . | . 7 .
4 . . | . . . | . . .
. . . | 8 5 7 | . . .
""", None)


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
        puzzle = Puzzle.from_string(examples['fiendish'].puzzle)
        self.assertTrue(puzzle.ready())
        self.assertEqual(str(puzzle), compact(examples['fiendish'].puzzle))

    def test_puzzle_is_consistent(self):
        self.assertTrue(Puzzle.from_string(examples['easy'].solution).is_consistent())
        non_consistent = '1' + examples['easy'].solution.strip()[1:]
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

    def test_chute_boxes(self):
        puzzle = Puzzle.from_string(examples['easy'].solution)
        chutes = [[box.id for box in chute] for chute_type, chute in puzzle.chutes_boxes()]
        self.assertEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8]], chutes)
        top_left_digit = [[box[0].digit for box in chute] for chute_type, chute in puzzle.chutes_boxes()]
        self.assertEqual([[8, 1, 3], [2, 6, 1], [6, 8, 5], [8, 2, 6], [1, 6, 8], [3, 1, 5]], top_left_digit)

class TestAssistedSolver(unittest.TestCase):
    def test_one_empty_square(self):
        grid = '.' + examples['easy'].solution.strip()[1:]
        puzzle = Puzzle.from_string(grid)
        self.assertTrue(run_assisted_solver(puzzle, False))

def solve_everything():
    for name in sorted(examples):
        puzzle = Puzzle.from_matrix(examples[name].puzzle)
        moves = run_assisted_solver(puzzle, False)
        if puzzle.solved():
            message = '%s: solved successfully in %d move(s)' % (name, moves)
        else:
            message = '%s: remains unsolved after %d move(s)' % (name, moves)
        print(message)

if __name__ == '__main__':
    solve_everything()

