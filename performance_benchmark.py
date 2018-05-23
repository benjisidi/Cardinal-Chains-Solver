if __name__ == '__main__':
	import timeit
	setup = '''
from Graph import Graph
t = [
	[4, 4, 4, 4, 4],
	[4, 2, 2, 2, 4],
	[4, 2, 0, 4, 4],
	[4, 2, 2, 4, 4],
	[4, 4, 4, 6, 8],
	[4, 4, 4, 4, 4],
	[4, 4, 4, 4, 4],
	[4, 4, 4, 4, 4],
	[4, 4, 4, 4, 4],
	[4, 4, 4, 4, 4],
	[4, 4, 4, 4, 4]]
g = Graph()
g.from_matrix(t)
g.check_for_end()
'''
	testBrute = '''
g.brute_force()
g.reset()	
'''
	testSimplify = '''
g.simplify()
g.brute_force()
g.reset()
'''
	its = 1
	repeats = 1
	print 'Average of {} iterations, best of {}:'.format(its, repeats)
	print '    Brute force: {:0.6f}s/iteration'.format(
		   min(timeit.repeat(testBrute, setup=setup, number=its, repeat=repeats))/its)
	print '    Simplify + brute force: {:0.6f}s/iteration'.format(
		   min(timeit.repeat(testSimplify, setup=setup, number=its, repeat=repeats))/its)