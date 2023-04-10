import numpy as np

from benchmark import BF

# Create an instance of the benchmark function
bf = BF("ackley")

# Calculate the function value at a given point
y = bf.evaluate(np.array([1, 1]))
print(y)

# Plot the function
bf.plot()
# or specify the number of points for the plot
bf.plot(num_points=32)

# Get the function itself
f = bf.get_function()
y = f(np.array([1, 1]))
print(y)



"""
You can also use the provided benchmark.py script to calculate and visualize benchmark functions from the command line.

> python benchmark.py --function ackley --x0 0 --x1 1

Or just visualize the function:

> python benchmark.py --function ackley
"""
