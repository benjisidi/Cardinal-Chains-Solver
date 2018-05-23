# import the necessary packages
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import pyautogui as pag
import pywinauto as pwa
import win32gui as w32
import mss
from Graph import Graph
from time import time

class Chains_OCR:
def extract_digits_and_symbols(image, charCnts, minW=3, minH=15):
	# grab the internal Python iterator for the list of character
	# contours, then  initialize the character ROI and location
	# lists, respectively
	print charCnts
	charIter = charCnts.__iter__()
	rois = []
	locs = []

	# keep looping over the character contours until we reach the end
	# of the list
	prev, current = None, charIter.next()
	while True:
		try:
			# grab the next character contour from the list, compute
			# its bounding box, and initialize the ROI
			c = next(charIter)
			(cX, cY, cW, cH) = cv2.boundingRect(c)
			roi = None

			# check to see if the width and height are sufficiently
			# large, indicating that we have found a digit
			if cW >= minW and cH >= minH:
				# extract the ROI
				roi = image[cY:cY + cH, cX:cX + cW]
				rois.append(roi)
				locs.append((cX, cY, cX + cW, cY + cH))

			# otherwise, we are examining one of the special symbols
			else:
				# MICR symbols include three separate parts, so we
				# need to grab the next two parts from our iterator,
				# followed by initializing the bounding box
				# coordinates for the symbol
				parts = [c, next(charIter), next(charIter)]
				(sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
				                        -np.inf)

				# loop over the parts
				for p in parts:
					# compute the bounding box for the part, then
					# update our bookkeeping variables
					(pX, pY, pW, pH) = cv2.boundingRect(p)
					sXA = min(sXA, pX)
					sYA = min(sYA, pY)
					sXB = max(sXB, pX + pW)
					sYB = max(sYB, pY + pH)

				# extract the ROI
				roi = image[sYA:sYB, sXA:sXB]
				rois.append(roi)
				locs.append((sXA, sYA, sXB, sYB))
			# we have reached the end of the iterator; gracefully break
			# from the loop
		except StopIteration:
			break

	# return a tuple of the ROIs and locations
	return (rois, locs)


def get_contours():
	# initialize the list of reference character names, in the same
	# order as they appear in the reference image
	charNames = ["x", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	# load the reference MICR image from disk, convert it to grayscale,
	# and threshold it, such that the digits appear as *white* on a
	# *black* background
	ref = cv2.imread("numbers_ref.png")
	ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
	ref = imutils.resize(ref, width=400)
	ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
	                    cv2.THRESH_OTSU)[1]
	ref = cv2.bitwise_not(ref)
	plt.imshow(ref)
	plt.show()

	# find contours in the MICR image (i.e,. the outlines of the
	# characters) and sort them from left to right
	refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	                           cv2.CHAIN_APPROX_SIMPLE)
	refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
	refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

	# create a clone of the original image so we can draw on it
	clone = np.dstack([ref.copy()] * 3)

	# loop over the (sorted) contours
	for c in refCnts:
		# compute the bounding box of the contour and draw it on our
		# image
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output of applying the simple contour method
	cv2.imshow("Simple Method", clone)
	cv2.waitKey(0)

	# extract the digits and symbols from the list of contours, then
	# initialize a dictionary to map the character name to the ROI
	(refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts,
	                                                minW=1, minH=15)
	chars = {}

	# re-initialize the clone image so we can draw on it again
	clone = np.dstack([ref.copy()] * 3)

	# loop over the reference ROIs and locations
	for (name, roi, loc) in zip(charNames, refROIs, refLocs):
		# draw a bounding box surrounding the character on the output
		# image
		(xA, yA, xB, yB) = loc
		cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# resize the ROI to a fixed size, then update the characters
		# dictionary, mapping the character name to the ROI
		roi = cv2.resize(roi, (30, 30))
		chars[name] = roi

		# display the character ROI to our screen
		cv2.imshow("Char", roi)
		cv2.waitKey(0)

	# show the output of our better method
	cv2.imshow("Better Method", clone)
	cv2.waitKey(0)


def resize_window():
	hwnd = w32.FindWindow(None, "Cardinal Chains")
	if hwnd == 0:
		print 'Could not find chains window.'
	else:
		w32.ShowWindow(hwnd, 5)
		w32.SetForegroundWindow(hwnd)
		w32.MoveWindow(hwnd, 100, 100, 1280, 960, False)
	return hwnd


def load_ref_files(sf=1):
	names = ['x', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	out = {}
	for name in names:
		img = cv2.imread('ref_files/' + name + '.png')
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (0, 0), fx=sf, fy=sf)
		_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
		out[name] = img
	return out



def get_screenshot(sf=1):
	with mss.mss() as sct:
		# The screen part to capture
		monitor = {'top': 100, 'left': 100, 'width': 1280, 'height': 960}
		output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)

		# Grab the data
		sct_img = sct.grab(monitor)
		sct_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2GRAY)
		mean = np.mean(sct_img)
		inv=mean > 180
		#sct_img= cv2.resize(sct_img, (0, 0), fx=sf, fy=sf)
		mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
		_, sct_img = cv2.threshold(sct_img, 100, 255, mode)
		sct_img = sct_img[50:940,100:1150]
		#cv2.imshow('',sct_img)
		#cv2.waitKey(0)
	return sct_img




def read_level():
	# Grab threshed image of chains level
	sct = get_screenshot()
	# Get inverted image
	inv = cv2.bitwise_not(sct)
	# Find connected components of both reg and inverted image
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sct)
	nlabelsI, labelsI, statsI, centroidsI = cv2.connectedComponentsWithStats(inv)

	# Drop two largest inverted image regions; background and dead space
	# Sort by component area
	statsI = statsI[statsI[:,4].argsort()]
	statsI = np.delete(statsI, -1, axis=0)
	statsI = np.delete(statsI, -1, axis=0)

	# Make the output array and coordinate conversion dictionaries
	array, CDX, CDY = matrix_from_points(statsI)
	# Drop the two components of the regular image; background and grid
	ordering = stats[:, 4].argsort()
	stats = stats[ordering]
	centroids = centroids[ordering]
	stats = np.delete(stats, -1, axis=0)
	stats = np.delete(stats, -1, axis=0)
	centroids = np.delete(centroids, -1, axis=0)
	centroids = np.delete(centroids, -1, axis=0)
	# Floor centroids to nearest 10 to group cols and rows
	centroids = np.floor(centroids/10.)*10

	stage1Classifier = {0:-1, 2:1, 3:7, 7:8, 4:None, 5:None, 6:None}
	stage2Classifier = {(1, 3, 1): None,
	                    (2, 2): 0,
						(3, 2): 3,
						(1, 1, 2): 4,
						(2, 3, 1): 6,
						(1, 3, 2): 9}
	for box in statsI:
		boxCenterX = box[0] + box[3]/2. +
		components = np.array([x for x in stats if (box[0] < x[0] < box[0] + box[2]) and
		                                           (box[1] < x[1] < box[1] + box[3])])
		class1 = stage1Classifier[len(components)]
		if class1 is not None:
			array[CDY[box[1]], CDX[box[0]]] = class1
			continue

		colValues = components[:,0]
		vals, counts = unique_within_tol(colValues, 8, return_counts=True)
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
	return array



def matrix_from_points(stats):
	# The unique x and y values of box TL points give us the layout of the matrix.
	# We round them to avoid offsets of 1px showing as "unique", and sort them
	# so we can make a dictionary of pixel coords to array coords
	xs = stats[:,0]
	ys = stats[:,1]
	uniqueXs = unique_within_tol(stats[:,0], 8)
	uniqueYs = unique_within_tol(stats[:,1], 8)
	uniqueXs.sort()
	uniqueYs.sort()

	conversionDictX = {}
	conversionDictY = {}
	for val in xs:
		conversionDictX[val] = find_nearest(uniqueXs, val)
	for val in ys:
		conversionDictY[val] = find_nearest(uniqueYs, val)
	xSize = len(uniqueXs)
	ySize = len(uniqueYs)
	array = np.full((ySize, xSize), -1)
	return array, conversionDictX, conversionDictY

# By @Guillaume S
# From https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
def unique_within_tol(array, tol, return_counts=False):
	if not return_counts:
		return np.unique(np.floor(array/tol).astype(int))*tol
	vals, counts = np.unique(np.floor(array/tol).astype(int), return_counts=True)
	return vals*tol, counts

# By @unutbu
# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
	idx = (np.abs(array - value)).argmin()
	return idx

def show_components(labels):
	# Map component labels to hue val
	label_hue = np.uint8(255 * labels / np.max(labels))
	blank_ch = 255 * np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue == 0] = 255

	cv2.imshow('labeled', labeled_img)
	cv2.waitKey()


if __name__ == '__main__':
	resize_window()
	t1 = time()
	matrix = read_level()
	g = Graph()
	g.from_matrix(matrix)
	g.check_for_end()
	g.simplify()
	g.brute_force()
	t2 = time()
	print g.solutions
	print 'Finished in {}s'.format(t2-t1)