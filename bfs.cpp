//Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS .
//g++ -fopenmp HPC.cpp -o HPC.exe
//.\HPC.exe

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// 1. Global graph avoids MinGW compiler crashes when passing vectors
vector<int> adj[8] = {
    {1, 2}, {0, 3, 4}, {0, 5, 6}, {1, 7}, {1}, {2}, {2}, {3}
};

int visitedDFS[8] = {0}; // Global visited tracker for DFS

// --- STANDARD PARALLEL BFS (Level-Order Double Buffering) ---
void parallelBFS(int start) {
    int visitedBFS[8] = {0};
    vector<int> currentFrontier; // Queue for the current level
    
    currentFrontier.push_back(start);
    visitedBFS[start] = 1;

    cout << "Parallel BFS: ";
    while (!currentFrontier.empty()) {
        vector<int> nextFrontier; // Queue for the next level
        
        // Process the current level in parallel safely
        #pragma omp parallel for
        for (int i = 0; i < currentFrontier.size(); i++) {
            int u = currentFrontier[i];
            
            #pragma omp critical
            cout << u << " ";

            for (int j = 0; j < adj[u].size(); j++) {
                int v = adj[u][j];
                if (visitedBFS[v] == 0) {
                    #pragma omp critical
                    {
                        if (visitedBFS[v] == 0) {
                            visitedBFS[v] = 1;
                            nextFrontier.push_back(v); // Safe push to the next level
                        }
                    }
                }
            }
        }
        currentFrontier = nextFrontier; // Move to the next depth level
    }
    cout << endl;
}

// --- STANDARD PARALLEL DFS (Task-Based) ---
void dfsRecursive(int u) {
    #pragma omp critical
    {
        if (visitedDFS[u] == 0) {
            visitedDFS[u] = 1;
            cout << u << " ";
        }
    }

    for (int i = 0; i < adj[u].size(); i++) {
        int v = adj[u][i];
        if (visitedDFS[v] == 0) {
            // "task" is the industry standard for parallel recursion
            #pragma omp task
            dfsRecursive(v);
        }
    }
}

void parallelDFS(int start) {
    cout << "Parallel DFS: ";
    #pragma omp parallel
    {
        // "single" ensures only one thread starts the recursion tree
        #pragma omp single
        dfsRecursive(start);
    }
    cout << endl;   
}

int main() {
    cout << "--- SPPU LP5: Standard & Correct Practical 1 ---" << endl;
    
    double start = omp_get_wtime();
    parallelBFS(0);
    cout << "BFS Time: " << (omp_get_wtime() - start) << " sec\n" << endl;

    start = omp_get_wtime();
    parallelDFS(0);
    cout << "DFS Time: " << (omp_get_wtime() - start) << " sec\n" << endl;

    return 0;
}



/*

Install the missing pthreads library (Quickest Fix)

Open the Start Menu and search for MinGW Installation Manager (or go to C:\MinGW\bin\mingw-get.exe and run it).

In the left panel, click on Basic Setup or All Packages -> MinGW -> MinGW Base System.

Scroll through the list on the right and look for packages named:

mingw32-pthreads-w32 (Look for the dev and bin classes)

Right-click on them and select Mark for Installation.

Click on the Installation menu at the top left, and select Apply Changes.

Once it finishes downloading, close your VS Code/PowerShell completely, reopen it, and run your compile command again:

*/