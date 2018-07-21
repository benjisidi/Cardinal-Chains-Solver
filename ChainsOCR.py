# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyautogui as pag
import win32gui as w32
import mss
from Graph import Graph
from time import time, sleep
import sys


class Chains_OCR:

	def __init__(self, debug=False):
		self.window_x = 100
		self.window_y = 100
		self.window_width = 1280
		self.window_height = 960
		self.sct_region = (50, 940, 100, 1150)
		self.debug = debug

	def resize_window(self,):
		hwnd = w32.FindWindow(None, "Cardinal Chains")
		if hwnd == 0:
			print 'Could not find chains window.'
			sys.exit(1)
		else:
			w32.ShowWindow(hwnd, 5)
			w32.SetForegroundWindow(hwnd)
			w32.MoveWindow(hwnd, self.window_x, self.window_y, self.window_width, self.window_height, False)
		return hwnd


	def load_ref_files(self, sf=1):
		names = ['x', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		out = {}
		for name in names:
			img = cv2.imread('ref_files/' + name + '.png')
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (0, 0), fx=sf, fy=sf)
			_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
			out[name] = img
		return out



	def get_screenshot(self):
		with mss.mss() as sct:
			# The screen part to capture
			monitor = {'top': self.window_y, 'left': self.window_x,
			           'width': self.window_width, 'height': self.window_height}

			# Grab the data and greyscale it
			sct_img = sct.grab(monitor)
			sct_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2GRAY)
			# Determine if we want to invert it or not (to deal with dark/light levels)
			mean = np.mean(sct_img)
			inv = mean > 180
			# Threshold the image
			mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
			_, sct_img = cv2.threshold(sct_img, 100, 255, mode)
			# Crop only the board area
			sct_img = sct_img[self.sct_region[0]:self.sct_region[1],
			                  self.sct_region[2]:self.sct_region[3]]
			# Show for debugging
			if self.debug:
				cv2.imshow('',sct_img)
				cv2.waitKey(0)
		return sct_img




	def read_level(self):
		# Grab threshed image of chains level
		sct = self.get_screenshot()
		# Get inverted image
		inv = cv2.bitwise_not(sct)
		# Find connected components of both reg and inverted image
		nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sct)
		nlabelsI, labelsI, statsI, centroidsI = cv2.connectedComponentsWithStats(inv)

		# Drop all large inverted image regions; background and dead space
		# Sort by component area
		statsI = statsI[statsI[:,4].argsort()]
		# Delete components with heights larger than the box height
		while statsI[-1,3] > 180:
			statsI = np.delete(statsI, -1, axis=0)

		# Sort regular image by component area
		stats = stats[stats[:, 4].argsort()]
		# Drop the two largest components of the regular image; background and grid
		stats = np.delete(stats, -1, axis=0)
		stats = np.delete(stats, -1, axis=0)

		# Make the output array and coordinate conversion dictionaries
		array, CDX, CDY = self.matrix_from_points(statsI)
		stage1Classifier = {0:-1, 2:1, 3:7, 7:8, 4:None, 5:None, 6:None}
		stage2Classifier = {(1, 3, 1): None,
		                    (2, 2): 0,
							(3, 2): 3,
							(1, 1, 2): 4,
							(2, 3, 1): 6,
							(1, 3, 2): 9}

		arrayToPix = {}
		for box in statsI:
			boxCenterX = box[0] + box[2]/2. + self.sct_region[2] + self.window_x
			boxCenterY = box[1] + box[3]/2. + self.sct_region[0] + self.window_y
			arrayX = CDX[box[0]]
			arrayY = CDY[box[1]]
			arrayToPix[(arrayY, arrayX)] = (boxCenterY, boxCenterX)
			components = np.array([x for x in stats if (box[0] < x[0] < box[0] + box[2]) and
			                                           (box[1] < x[1] < box[1] + box[3])])

			class1 = stage1Classifier[len(components)]
			if class1 is not None:
				array[CDY[box[1]], CDX[box[0]]] = class1
				continue

			colValues = components[:,0]
			vals, counts = self.unique_within_tol(colValues, 8, return_counts=True)
			class2 = stage2Classifier[tuple(counts)]
			if class2 is not None:
				array[CDY[box[1]], CDX[box[0]]] = class2
				continue

			rowValues = components[:,1]
			rowIndicies = np.argsort(rowValues)
			colValuesSorted = colValues[rowIndicies]

			if colValuesSorted[0] < colValuesSorted[1]:
				array[CDY[box[1]], CDX[box[0]]] = 2
			else:
				array[CDY[box[1]], CDX[box[0]]] = 5
		return array, arrayToPix



	def matrix_from_points(self, stats):
		# The unique x and y values of box TL points give us the layout of the matrix.
		# We round them to avoid offsets of 1px showing as "unique", and sort them
		# so we can make a dictionary of pixel coords to array coords
		xs = stats[:,0]
		ys = stats[:,1]
		uniqueXs = self.unique_within(stats[:,0], 8)
		uniqueYs = self.unique_within(stats[:,1], 8)
		uniqueXs.sort()
		uniqueYs.sort()
		conversionDictX = {}
		conversionDictY = {}
		for val in xs:
			conversionDictX[val] = self.find_nearest(uniqueXs, val)
		for val in ys:
			conversionDictY[val] = self.find_nearest(uniqueYs, val)
		xSize = len(uniqueXs)
		ySize = len(uniqueYs)
		array = np.full((ySize, xSize), -1)
		return array, conversionDictX, conversionDictY

	# By @Guillaume S
	# From https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
	def unique_within_tol(self, array, tol, return_counts=False):
		if not return_counts:
			return np.unique(np.floor(array/tol).astype(int))*tol
		vals, counts = np.unique(np.floor(array/tol).astype(int), return_counts=True)
		return vals*tol, counts

	def unique_within(self, array, tol, return_counts=False):
		arrayCopy = array.copy()
		uniqueVals = []
		counts = []
		while len(arrayCopy) > 0:
			val = arrayCopy[0]
			np.delete(arrayCopy, 0)
			uniqueVals.append(val)
			count = 1
			mask = np.bitwise_not(abs(arrayCopy - val) < tol)
			arrayCopy = arrayCopy[mask,...]
			count += np.sum(mask)
			counts.append(count)
		if return_counts:
			return uniqueVals, counts
		return uniqueVals


	def find_nearest(self, array, value):
		idx = (np.abs(array - value)).argmin()
		return idx

	def show_components(self, labels):
		# Map component labels to hue val
		label_hue = np.uint8(255 * labels / np.max(labels))
		blank_ch = 255 * np.ones_like(label_hue)
		labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

		# cvt to BGR for display
		labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

		# set bg label to black
		labeled_img[label_hue == 0] = 0

		cv2.imshow('labeled', labeled_img)
		cv2.waitKey()

	def play_level(self, solution, dict):
		for i, pt in enumerate(solution):
			sq = dict[pt]
			pag.moveTo(sq[1], sq[0], 0)
			if i == 0:
				pag.mouseDown()
			elif i == len(solution) -1:
				pag.mouseUp()


	def solve_all_levels(self):
		t1 = time()
		self.resize_window()
		for i in range(16):
			matrix, matrixToPix = c.read_level()
			g = Graph()
			g.from_matrix(matrix)
			g.simplify()
			g.check_for_end()
			g.brute_force()
			self.play_level(g.solutions[0], matrixToPix)
			pag.moveTo(1304, 448, 0)
			pag.mouseDown()
			pag.mouseUp()
			sleep(1)
		t2 = time()
		print 'Finished in {}s'.format(t2 - t1)

	def solve_current_level(self):
		self.resize_window()
		matrix, matrixToPix = c.read_level()
		g = Graph()
		g.from_matrix(matrix)
		if self.debug:
			g.draw()
		g.check_for_end()
		g.convert_endpoint_vertices()
		g.simplify()
		if self.debug:
			g.draw()
		g.brute_force()
		self.play_level(g.solutions[0], matrixToPix)


if __name__ == '__main__':
	c = Chains_OCR()
	#c.solve_all_levels()
	c.solve_current_level()