__author__ = 'Zhang'
import cv2
import numpy as np

CROSS = [
	(-1, 0),
	(1, 0),
	(0, 1),
	(0, -1)
]

def getBOX(n):
	BOX = []
	x = -n
	while x <= n:
		y = -n
		while y <= n:
			BOX.append((x,y))
			y += 1
		x += 1
	return BOX

def inpaint(image, mask):
	width = image.shape[0]
	height = image.shape[1]
	dst = np.ones((width, height), np.uint8)
	KNOWN = 0
	INSIDE = 255
	BAND = 1
	F = [0] * width * height
	T = [0] * width * height
	narrow_band = PriorityQueue()

#Get the index of the point
	def index(x, y):
		return y * width + x

#Judge if the point is valid
	def is_valid(bx, by):
		return bx >=0 and by >= 0 and bx < width and by < height

	def _inpaint(x, y):
		b_e = []
		for d in BOX:
			if d[0] == 0 and d[1] == 0:
				continue
			nx = x + d[0]
			ny = y + d[1]
			if not is_valid(nx, ny):
				continue
			ni = index(nx, ny)
			if F[ni] == KNOWN and image[nx, ny] != 255 and image[nx, ny] != 0:
				b_e.append(image[nx, ny])
		if len(b_e) != 0:
			color = sum(b_e) / len(b_e)
			dst[x, y] = color

#Funnction Solve
	def _solve(x1, y1, x2, y2):
		p1_valid = is_valid(x1, y1)
		p2_valid = is_valid(x2, y2)
		i1 = index(x1, y1)
		i2 = index(x2, y2)
		sol = float("inf")
		if p1_valid and F[i1] == KNOWN:
			t1 = T[i1]
			if p2_valid and F[i2] == KNOWN:
				t2 = T[i2]
				r = np.sqrt(2 * (t1 - t2) * (t1 - t2))
				s = (t1 + t2 * r) / 2
				if s >= t1 and s >= t2:
					sol = s
				else:
					s += r
					if s >= t1 and s >= t2:
						sol = s
			else:
				sol = 1 + t1
		elif p2_valid and F[i2] == KNOWN:
			t2 = T[i2]
			sol = 1 + t2
		return sol

#Using mask to get the bound and inside
	for y in range(height):
		for x in range(width):
			i = index(x, y)
			flag = KNOWN
			if mask[x, y] == 255:
				flag = INSIDE
			dist = 0.0
			n_total = 0
			n_unknown = 0
			if flag != KNOWN:
				for d in CROSS:
					nx = x + d[0]
					ny = y + d[1]
					if not is_valid(nx, ny):
						continue
					n_total += 1
					if mask[nx, ny] == 255:
						n_unknown += 1
				if n_total > 0 and n_total == n_unknown:
					flag = INSIDE
					dist = float("inf")
				else:
					flag = BAND
					_inpaint(x, y)
					narrow_band.push(dist, (x, y, i))
			F[i] = flag
			T[i] = dist

	while not narrow_band.empty():
		c = narrow_band.pop()
		if c == None:
			continue
		F[c[2]] = KNOWN
		for d in CROSS:
			nx = c[0] + d[0]
			ny = c[1] + d[1]
			if not is_valid(nx, ny):
				continue
			ni = index(nx, ny)
			if F[ni] != KNOWN:
				T[ni] = min([
					_solve(nx - 1, ny, nx, ny - 1),
					_solve(nx + 1, ny, nx, ny - 1),
					_solve(nx - 1, ny, nx, ny + 1),
					_solve(nx + 1, ny, nx, ny + 1)
				])
				if F[ni] == INSIDE:
					F[ni] = BAND
					_inpaint(nx, ny)
					narrow_band.push(T[ni], (nx, ny, ni))

	for y in range(height):
		for x in range(width):
			if mask[x, y] == 0:
				dst[x, y] = image[x, y]

	return dst

import heapq
import itertools

class PriorityQueue:
	def __init__(self):
		self._q = []
		self._entry_map = {}
		self._counter = itertools.count()

	def push(self, priority, task):
		if task in self._entry_map:
			self.remove(task)
		count = next(self._counter)
		entry = [priority, count, task]
		self._entry_map[task] = entry
		heapq.heappush(self._q, entry)

	def remove(self, task):
		entry = self._entry_map.pop(task)
		entry[-1] = None

	def pop(self):
		while self._q:
			p, c, task = heapq.heappop(self._q)
			if task:
				del self._entry_map[task]
				return task
		return

	def empty(self):
		return len(self._q) == 0

if __name__ == '__main__':
#You can change the paths and box size here.
	img = cv2.imread('../inputs/3DamagedBeach.jpg')
	mask = cv2.imread('../inputs/3Mask.jpg', 0)
	box_size = 5

#The main process start here
	BOX = getBOX(box_size)
	cv2.imshow("Damaged Image", img)
	width = img.shape[0]
	height = img.shape[1]
	b,g,r = cv2.split(img)
	mask2 = np.ones((width,height), np.uint8)
	for i in range(width):
		for j in range(height):
			mask2[i][j] = 255 - mask[i][j]
			if mask2[i][j] == 0:
				continue
			else:
				mask2[i][j] = 255
	cv2.imshow("Mask", mask2)
	print "Inpainting Blue Layer"
	dst_b = inpaint(b, mask2)
	print "Inpainting Green Layer"
	dst_g = inpaint(g, mask2)
	print "Inpainting Red Layer"
	dst_r = inpaint(r, mask2)
	print "Finish"
	dst = cv2.merge((dst_b,dst_g,dst_r))
	cv2.imwrite("../outputs/output.jpg", dst)
	cv2.imshow("Result", dst)
	cv2.waitKey(0)