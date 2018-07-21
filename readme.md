# Cardinal Chains Solver

This is an ongoing project to develop and implement an algorithm to computationally solve the puzzle game [*Cardinal Chains*](https://danijmn.itch.io/cardinalchains), made by Daniel Nora.

The game essentially boils down to solving the Hamiltonian Path problem, with some additional simplifying and complicating features.

A long-form write-up of the development of this project is available on my blog, [here](http://insomniaccoder.com/2018/05/10/computationally-solving-cardinal-chains-an-adventure-with-graphs/).

## Usage

- Download the *Cardinal Chains* demo (or buy the game, it's fantastic) from [here](https://danijmn.itch.io/cardinalchains). 
- Download the project directory
- Open *Cardinal Chains* and navigate to the level you wish to solve (Currently solves 1-20 excluding 10)
- Run *ChainsOCR.py*

## File Structure

| File/Folder              | Contents                                                     |
| ------------------------ | ------------------------------------------------------------ |
| Graph.py                 | Holds the Graph object, which stores the graph representation of each level, along with methods to construct the graph from a 2d matrix, display it in a Tk window, and simplify and solve it. |
| ChainsOCR.py             | Holds the ChainsOCR object, which contains methods to "read" each level from the game window and move the mouse once a solution has been computed by *Graph*. |
| graph_tester.py          | Simple script for testing the Graph object's *draw* function. |
| performance_benchmark.py | Script comparing solving a level by attempting some logical simplifications of the graph, and simply brute-forcing it. Results are shown in the aforementioned blog post. |



## Future Improvements

- Known issue in failing to solve level 10 due to not being able to identify an end-point.

- Currently can't solve multi-coloured levels. This will require significant work.