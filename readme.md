# Cardinal Chains Solver

This is an ongoing project to develop and implement an algorithm to computationally solve the puzzle game [*Cardinal Chains*](https://danijmn.itch.io/cardinalchains), made by Daniel Nora.

The game essentially boils down to solving the Hamiltonian Path problem, with some additional simplifying features.

## Usage

- Download the *Cardinal Chains* demo (or buy the game, it's fantastic) from [here](https://danijmn.itch.io/cardinalchains). 
- Download the project directory
- Open *Cardinal Chains* and navigate to the level you wish to solve (Currently solves 1-20 excluding 10)
- Run *ChainsOCR.py*

#### Dependencies

- [NumPy](http://www.numpy.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)
- [win32GUI](https://pypi.org/project/win32gui/)
- [mss](https://pypi.org/project/mss/)
- [OpenCV](https://pypi.org/project/opencv-python/)

## File Structure

| File/Folder              | Contents                                                     |
| ------------------------ | ------------------------------------------------------------ |
| Graph.py                 | Holds the Graph object, which stores the graph representation of each level, along with methods to construct the graph from a 2d matrix, display it in a Tk window, and simplify and solve it. |
| ChainsOCR.py             | Holds the ChainsOCR object, which contains methods to "read" each level from the game window and move the mouse once a solution has been computed by *Graph*. |
| graph_tester.py          | Simple script for testing the Graph object's *draw* function. |
| performance_benchmark.py | Script comparing solving a level by attempting some logical simplifications of the graph, and simply brute-forcing it. |

I am debugging pull requests and need a change to merge.


## Future Improvements

- Known issue in failing to solve level 10 due to not being able to identify an end-point.

- Currently can't solve multi-coloured levels. This will require significant work.
