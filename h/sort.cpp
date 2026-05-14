// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms.
//g++ -fopenmp HPC.cpp -o HPC.exe
//.\HPC.exe

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// --- Bubble Sorts ---
void sequentialBubble(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
}

void parallelBubble(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2)
            if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
    }
}

// --- Merge Sorts ---
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r) temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (int p = 0; p < k; p++) arr[l + p] = temp[p];
}

void sequentialMerge(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        sequentialMerge(arr, l, m);
        sequentialMerge(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMerge(vector<int>& arr, int l, int r, int depth = 0) {
    if (l < r) {
        int m = l + (r - l) / 2;
        if (depth < 2) { 
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMerge(arr, l, m, depth + 1);
                #pragma omp section
                parallelMerge(arr, m + 1, r, depth + 1);
            }
        } else {
            sequentialMerge(arr, l, m);
            sequentialMerge(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    int n = 10000;
    vector<int> a1(n), a2(n), a3(n), a4(n);
    for (int i = 0; i < n; i++) a1[i] = a2[i] = a3[i] = a4[i] = rand() % 100000;

    cout << "--- SPPU LP5: Practical 2 (Array Size: " << n << ") ---\n";
    
    double start = omp_get_wtime();
    sequentialBubble(a1);
    cout << "Sequential Bubble : " << omp_get_wtime() - start << " sec\n";

    start = omp_get_wtime();
    parallelBubble(a2);
    cout << "Parallel Bubble   : " << omp_get_wtime() - start << " sec\n";

    start = omp_get_wtime();
    sequentialMerge(a3, 0, n - 1);
    cout << "Sequential Merge  : " << omp_get_wtime() - start << " sec\n";

    start = omp_get_wtime();
    omp_set_nested(1);
    parallelMerge(a4, 0, n - 1);
    cout << "Parallel Merge    : " << omp_get_wtime() - start << " sec\n";

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