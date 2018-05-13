from tkinter import *
from tkinter import ttk
from tkinter import font
import sys

class Graph:
    def __init__(self):
        self.nodes = {} # Nodes are stored as a dict with their ids as keys
        self.endPoints = []
        self.startPoint = None

    # Check if there's a two-way vertex between two nodes
    def twoWay(self, x, y):
        return ((y in self.nodes[x]) and (x in self.nodes[y]))


    def from_matrix(self, matrix):
        self.matrix = matrix
        for rowI, row in enumerate(matrix):
            for colI, val in enumerate(row):
                # If cell is empty, do nothing
                if val == -1:
                    pass    
                # Otherwise, check all 4 neighbours (if they exist)
                # and add the relevant vetices to the graph
                else:
                    if val == 0:
                        self.startPoint = val
                    self.nodes[(rowI, colI)] = []
                    for (i, j) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        # Column out of bounds
                        if (rowI + i >= 0 and rowI + i < len(matrix) and  
                            # Row out of bounds
                            colI + j >= 0 and colI + j < len(matrix[0]) and 
                            # Comparing node with itself
                            not (i == 0 and j == 0)): 
                            if matrix[rowI + i][colI + j] >= val:
                                self.nodes[(rowI, colI)].append((rowI+i, colI+j))                          

    def draw(self):
        # Window width and height, and matrix box side length in pixels
        width = 1280
        height = 720
        box = 100
        fontSize = 36
        # Set up a simple Tk canvas
        root = Tk()
        c = Canvas(root, width=width, height=height)
        c.pack(side="top", fill="both", expand=True)

        # Add headings and divider to our display
        c.create_text(width/4., height*.1, text="Level Matrix", font=("TkDefaultFont",fontSize))
        c.create_text(3*width/4., height*.1, text="Level Graph", font=("TkDefaultFont",fontSize))
        c.create_line(width/2., 0, width/2., height)


        # Now we're going to display the input matrix and
        # the generated graph, side by side
        # We want these to sit nicely centered in their halves,
        # so first we're going to calculate how much space we need
        # for our representation
        diagramWidth = box*len(self.matrix[0])
        diagramHeight = box*len(self.matrix)
        padX = (width/2. - diagramWidth)/2.
        padY = (height - diagramHeight)/2.

        # Make the matrix representation
        for rowI, row in enumerate(self.matrix):
            for colI, val in enumerate(row):
                # Calculate rectangle placement and draw it
                rectX = padX + colI * box
                rectY = padY + rowI * box
                c.create_rectangle(rectX, rectY, rectX + box, rectY + box, outline="black")
                # Apply the relevant label, if any
                if val != -1:
                    if val == 0:
                        c.create_text(rectX + box/2., rectY + box/2., text='S', font=("TkDefaultFont",fontSize))
                    else:
                        c.create_text(rectX + box/2., rectY + box/2., text=str(val), font=("TkDefaultFont",fontSize))

        # Make the graph representation
        # We need to shift accross to the other section of the window
                r = box * .1 # Radius of the node dots
                rectX += width/2. + box/2. + r/2.
                rectY += box/2. + r/2.
                sep = (box - 2*r*1.4) # This is the length of the vertex arrows
                # Draw a dot for each node
                if val != -1:
                    c.create_oval(rectX - r, rectY - r, rectX + r, rectY + r, fill='black')
                    # For each vertex, draw the relevant arrow
                    if (rowI, colI) in self.nodes.keys():
                        for vertex in self.nodes[(rowI, colI)]:
                            direction = (vertex[0] - rowI, vertex[1] - colI)
                            startX = rectX  + direction[1]*r*1.4
                            startY = rectY  + direction[0]*r*1.4
                            endX = startX + direction[1]*sep
                            endY  = startY + direction[0]*sep
                            c.create_line(startX, startY, endX, endY, arrow="last", width=3, arrowshape=(16,20,6))

        def quit_window(event):
            root.destroy()

        root.bind('<Escape>', quit_window)
        root.mainloop()


    # Rule 1
    def check_for_end(self):
        endPoints = []
        for node in self.nodes:
            # Grab all the vertices LEAVING the node in question
            vertices = self.nodes[node]
            # No exits at all
            if (vertices == [] or (
            # Just one two-way node that therefore must be entrance
            len(vertices) == 1 and self.twoWay(node, vertices[0])) and self.nodes.values().count(node) == 1):
                endPoints.append(node)
        self.endPoints = endPoints        
        return endPoints

    # Rule 2 - If a node with a two-way vertex is an end point, the vertex can be made into an entry vertex.
    # i.e. end points shouldn't have any exit nodes
    def convert_endpoint_vertices(self):
        for node in self.endPoints:
                self.nodes[node] = []


    # Rules 3 - 6
    def check_vertices(self):
        for node in self.nodes:
            # List of nodes that have exits that lead to this node
            enteringNodes = [x for x in self.nodes.keys() if node in self.nodes[x]]

            # Rule 3 - If a node has only a two-way vertex and exit vertices, 
            # the two-way vertex can be converted to an entry vertex
            if len(enteringNodes) == 1 and self.twoWay(node, enteringNodes[0]):
                self.nodes[node].remove(enteringNodes[0])

            # Rule 4 - If a node has only a two-way vertex and entry vertices, 
            # the two-way vertex can be converted to an exit vertex 
            if len(self.nodes[node]) == 1 and self.twoWay(node, self.nodes[node][0]):
                self.nodes[self.nodes[node][0]].remove(node)

            # Rule 5 - If a node is part of the only path to an end point, 
            # that node can't have any other exits
            if self.endPoints != []:
                # First we need to build a list of the path to the end point
                path = []
                currentNode = self.endPoints[0]
                currentEnteringNodes = [x for x in self.nodes.keys() if currentNode in self.nodes[x]]
                while len(currentEnteringNodes) == 1:
                    path.append(currentNode)
                    currentNode = currentEnteringNodes[0]
                    currentEnteringNodes = [x for x in self.nodes.keys() if currentNode in self.nodes[x]]
                # Now we check whether any of this node's exits lead to the path
                onlyValidExit = None
                for exit in self.nodes[node]:
                    if exit in path:
                        onlyValidExit = exit
                # If that's the case, we then remove all the remaining exits.                       
                if onlyValidExit is not None:
                    print onlyValidExit
                    self.nodes[node] = [onlyValidExit]
        
            # Rule 6 - If a node's only exit leads to an end,
            # all other entrances to that end can be destroyed
            # --- REPLACED WITH ---
            # Rule 7 - If a node has only one exit, all other entrances 
            # to its destination node can be removed. WRITE THIS UP!
            if len(self.nodes[node]) == 1:
                # Need to grab all the nodes that lead to the destination in question
                penultimateNodes = [x for x in self.nodes if self.nodes[node][0] in self.nodes[x]]
                # Now iterate over that list and remove the destination from the other nodes vertices
                for pNode in penultimateNodes:
                    if pNode != node:
                        self.nodes[pNode].remove(self.nodes[node][0])


    # Brute force method:
    # Use depth-first-search with backtracking.
    def brute_force(self):
        branches = []
        curNode = self.startPoint
        finished = False
        for node in self.nodes[self.startPoint]:
            visited = []


    def visit(self, node, path_=[], counter=0):
        path = path_
        for dest in self.nodes[node]:
            path.append(dest)
            path.append(self.visit(dest))
            counter += 1
        return path

'''
  a--b--c
 /
s
 \
  d--e--f


want: [(s), (s, a), (s, a, b), (s, a, b, c), (s, d), (s, d, e), (s, d, e, f)]
Start with empty master list
for each destination:
    add list containing path to that destination to master list

'''