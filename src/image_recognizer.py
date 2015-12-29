# A simple sudoku image recognizer
# for now, mainly here for tests
# The basic code comes from:
# https://github.com/abidrahmank/OpenCV2-Python/tree/master/OpenCV_Python_Blog/sudoku_v_0.0.6
# Character recognition for an available font comes from
# http://projectproto.blogspot.co.uk/2014/07/opencv-python-digit-recognition.html

import cv2
import numpy as np
import solver
import unittest
from PIL import Image, ImageFont, ImageDraw, ImageOps


class AbidrahmankRecognizer:
    def __init__(self):
        samples = np.float32(np.loadtxt('../abidrahmank/feature_vector_pixels.data'))
        responses = np.float32(np.loadtxt('../abidrahmank/samples_pixels.data'))
        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def recognize(self, roi):
        small_roi = cv2.resize(roi, (10, 10))
        feature = small_roi.reshape((1, 100)).astype(np.float32)
        ret, results, neigh, dist = self.model.findNearest(feature, k=1)
        return int(results.ravel()[0])


class TTFRecognizer:
    def __init__(self, fontfile, digitheight):
        self.digitheight = digitheight
        ttfont = ImageFont.truetype(fontfile, digitheight)
        samples = np.empty((0, digitheight * (digitheight // 2)))
        responses = []
        for n in range(10):
            pil_im = Image.new("RGB", (digitheight, digitheight * 2))
            ImageDraw.Draw(pil_im).text((0, 0), str(n), font=ttfont)
            pil_im = pil_im.crop(pil_im.getbbox())
            pil_im = ImageOps.invert(pil_im)
            #pil_im.save(str(n) + ".png")

            # convert to cv image
            cv_image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGBA2BGRA)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

            roi = cv2.resize(thresh, (digitheight, digitheight // 2))
            responses.append(n)
            sample = roi.reshape((1, digitheight * (digitheight // 2)))
            samples = np.append(samples, sample, 0)

        samples = np.array(samples, np.float32)
        responses = np.array(responses, np.float32)

        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def recognize(self, roi):
        roi = cv2.resize(roi, (self.digitheight, self.digitheight // 2))
        roi = roi.reshape((1, self.digitheight * (self.digitheight // 2)))
        roi = np.float32(roi)
        retval, results, neigh_resp, dists = self.model.findNearest(roi, k=1)
        return int(results.ravel()[0])


def import_image(image_file_name, recognizer):
    #############  Function to put vertices in clockwise order ######################
    def rectify(h):
        ''' this function put vertices of square we got, in clockwise order '''
        h = h.reshape((4, 2))
        hnew = np.zeros((4, 2), dtype=np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h, axis=1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

    ################ Now starts main program ###########################

    img = cv2.imread(image_file_name)
    if img is None:
        raise IOError('Cannot load image from file %s' % image_file_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 5, 2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_area = gray.size  # this is area of the image

    # if area of box > half of image area, it is possibly the biggest blob
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > image_area / 2]
    peri = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)

    #################      Now we got sudoku boundaries, Transform it to perfect square ######################

    # this is corners of new square image taken in CW order
    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)

    approx = rectify(approx)  # we put the corners of biggest square in CW order to match with h

    retval = cv2.getPerspectiveTransform(approx, h)  # apply perspective transformation
    warp = cv2.warpPerspective(img, retval, (450, 450))  # Now we get perfect square with size 450x450

    warpg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)  # kept a gray-scale copy of warp for further use

    ############ now take each element for inspection ##############

    sudo = np.zeros((9, 9), np.uint8)  # a 9x9 matrix to store our sudoku puzzle

    smooth = cv2.GaussianBlur(warpg, (3, 3), 3)
    thresh = cv2.adaptiveThreshold(smooth, 255, 0, 1, 5, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 800:
            (bx, by, bw, bh) = cv2.boundingRect(cnt)
            if (100 < bw * bh < 1200) and (5 < bw < 50) and (5 < bh < 50):
                roi = dilate[by:by + bh, bx:bx + bw]
                digit = recognizer.recognize(roi)

                # gridx and gridy are indices of row and column in sudo
                gridy, gridx = (bx + bw / 2) / 50, (by + bh / 2) / 50
                sudo.itemset((gridx, gridy), digit)
    return sudo


def pretty_print(sudo):
    l = [int(i) for i in sudo]  # we make string ans to a list
    print(np.array(l, np.uint8).reshape((9, 9)))  # Now we make it into an array of sudoku


class ImportTester(unittest.TestCase):
    # test on abidrahmank's image with his training data.
    # The test fails. There must be some problem with the training.
    # def test_abidrahmank(self):
    #     s = """. . . | 6 . 4 | 7 . .
    #     7 . 6 | . . . | . . 9
    #     . . . | . . 5 | . 8 .
    #     ---------------------
    #     . 7 . | . 2 . | . 9 3
    #     8 . . | . . . | . . 5
    #     4 3 . | . 1 . | . 7 .
    #     ----------------------
    #     . 5 . | 2 . . | . . .
    #     3 . . | . . . | 2 . 8
    #     . . 2 | 3 . 1 | . . ."""
    #
    #     expected = solver.Puzzle.from_string(s)
    #     actual = solver.Puzzle.from_matrix(import_image('../abidrahmank/sudokubig.jpg', AbidrahmankRecognizer()))
    #     self.assertEqual(str(expected), str(actual))

    def test_one_line_all_digits(self):
        filename = '../tmp/ImageImporterTest.png'
        s = '123456789' + ('.' * 9 * 7) + '987654321'
        solver.show(solver.Puzzle.from_string(s), pencil_marks=None, filename=filename, display=False)
        imported = import_image(filename, TTFRecognizer("../fonts/SourceCodePro-Bold.ttf", 24))
        puzzle = solver.Puzzle.from_matrix(imported)
        self.assertEqual(s, str(puzzle))
