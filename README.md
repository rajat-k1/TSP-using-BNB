# Python Script for Solving the Traveling Salesman Problem (TSP)

## Description
- This script is designed to solve the Traveling Salesman Problem using the Branch and Bound method. 
- It reads TSP data from text files, processes each file to find the minimum cost path, and writes the results to a CSV file.

## Requirements
- Python 3.x
- NumPy library

## Installation
Ensure that Python 3 is installed on your system. You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install NumPy using pip:

```pip install numpy```

## Usage
1. Place your TSP data files in a directory.
2. Open the script and modify the file path in the `__main__` section to point to your directory containing the TSP data files.
3. Run the script using Python:
```python your_script_name.py``` (Here it is TSPusingbnb.py)

The script will process each file and write the results to 'tsp_results_bnb.csv' in the current directory.

## Important Notes
- The script uses a Branch and Bound algorithm, which may take significant time for large datasets.
- I have mentioned ```#TODO``` in the script wherever changes are required.
- A timeout mechanism is implemented to stop processing a file if it takes too long, you can change the duration of this duration in the `__main__`. 
    (I have taken 100 milliseconds)
- Change the value in the for loop in the `__main__` section according to the number of data files you have.
- Modify the `file_path` variable in the `__main__` section according to the location of your test data.
- The script creates a 'tsp_results_bnb.csv' file to store the results. Ensure you have write permissions in the directory where the script is run.
