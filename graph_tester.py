from Graph import Graph
t = [
	[1, 1,  0],
	[ 1,  1,  1],
	[ 1,  1,  1],
	[ 2,  2,  3]]

g = Graph()
g.from_matrix(t)
g.check_for_end()
g.draw()
for i in range(0, 12):
	g.check_vertices()
	g.draw()

n = g.check_for_end()
print n