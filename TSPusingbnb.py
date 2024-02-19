import numpy as np
from queue import PriorityQueue
import signal
import time
import csv
import os
import atexit

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)

#function for reading the input files for solving TSP
#You can tweak the code if your input file format is different.
#I have attached some sample for input files in the repo
def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    
    size = int(lines[0].strip())
    matrix = np.zeros((size, size))


    for i in range(1, size + 1):
        matrix[i - 1] = [float(x) for x in lines[i].strip().split()]


    return matrix

#cleanup after writing to CSV
def cleanup():
    global file
    file.close()
    print("CSV file closed successfully")


#########################################################################################################################################
class TravelingSalesmanProblem:
    def __init__(self, graph):
        self.graph = np.array(graph)
        self.N = len(graph)
        self.INF = float('inf')

    #Logic to reduce the matrix
    def reduce_matrix(self, matrix, excluded_rows=None, excluded_cols=None):
        if excluded_rows is None:
            excluded_rows = []
        if excluded_cols is None:
            excluded_cols = []


        row_min = np.min(matrix, axis=1)
        row_min[excluded_rows] = 0
        matrix = matrix - row_min[:, np.newaxis]


        col_min = np.min(matrix, axis=0)
        col_min[excluded_cols] = 0
        matrix = matrix - col_min


        reduction_cost = np.sum(row_min) + np.sum(col_min)
        return matrix, reduction_cost


    def TSPRec(self, curr_bound, curr_weight, level, curr_path, visited, matrix):
        if level == self.N:
            if self.graph[curr_path[level - 1]][curr_path[0]] != self.INF:
                curr_res = curr_weight + self.graph[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.min_cost:
                    self.final_path = np.copy(curr_path)
                    self.final_path[self.N] = curr_path[0]
                    self.min_cost = curr_res
            return


        pq = PriorityQueue()
        for i in range(self.N):
            if self.graph[curr_path[level - 1]][i] != self.INF and not visited[i]:
                temp_bound = curr_bound
                temp_weight = curr_weight
                temp_matrix = np.copy(matrix)
                
                
                # Adjusting the bound based on the updated matrix
                min_val = np.min(temp_matrix[curr_path[level - 1]])
                temp_bound -= min_val
                temp_matrix[curr_path[level - 1]] -= min_val
                temp_matrix[:, i] -= min_val


                # Adjusting the bound based on the weight and matrix values
                temp_bound += (temp_weight + self.graph[curr_path[level - 1]][i]) / 2


                if temp_bound + temp_matrix.sum() < self.min_cost:
                    pq.put((temp_bound, i, temp_weight, temp_matrix))


        while not pq.empty():
            temp_bound, i, temp_weight, temp_matrix = pq.get()
            curr_path[level] = i
            visited[i] = True


            reduced_matrix, _ = self.reduce_matrix(temp_matrix,
                                                  excluded_rows=[curr_path[j] for j in range(level)],
                                                  excluded_cols=[i])
            self.TSPRec(temp_bound, temp_weight + self.graph[curr_path[level - 1]][i], level + 1, curr_path, visited, reduced_matrix)


            curr_path[level] = -1
            visited[i] = False


    def TSP(self):
        reduced_matrix, cost = self.reduce_matrix(self.graph)
        self.min_cost = float('inf')
        self.final_path = [-1] * (self.N + 1)
        curr_path = [-1] * (self.N + 1)
        visited = [False] * self.N
        curr_path[0] = 0
        visited[0] = True
        self.TSPRec(cost, 0, 1, curr_path, visited, reduced_matrix)
        return self.min_cost, self.final_path
#########################################################################################################################################
    

#Printing the path for easier understanding of the path taken
def print_solution(cost, path):
    print("Minimum cost:", cost)
    print("Path taken:", '->'.join(map(str, path)))

#Running below code if the script is run as the main program
if __name__ == "__main__":
    
    # Writing to CSV
    with open('tsp_results_bnb.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        atexit.register(cleanup)

        
        for i in range(1,101): #TODO Change the range of the loop based on your number of files, I had 100 files.
            # TODO Replace with actual path in your system in the next line
            #example: file_path = f'/Users/rajat/TSPData/tsp{i}.txt'
            file_path = f'some path'
            file_name = os.path.basename(file_path)
            start_time = time.time()  # Start the timer

            try:
                signal.setitimer(signal.ITIMER_REAL, 600)  # TODO Set the timeout accrding to your needs; I set it to duration 100 milliseconds
                graph = read_tsp_file(file_path)

                tsp = TravelingSalesmanProblem(graph)
                cost, path = tsp.TSP()

                # End the timer
                end_time = time.time()
                # Calculate elapsed time
                elapsed_time = end_time - start_time

                # Write the result for each file immediately
                writer.writerow([file_name, cost, elapsed_time])

                print(f"Solution for {file_path}:")
                print_solution(cost, path)
                print(f"Time taken: {elapsed_time:.2f} seconds")

            #Handling timeout
            except TimeoutException:
                writer.writerow([file_name, "N/A", "Timeout", ])
                print(f"Processing {file_path} timed out.")
            # Disable the alarm
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)