// Implement Min, Max, Sum and Average operations using Parallel Reduction.
//g++ -fopenmp HPC.cpp -o HPC.exe
//.\HPC.exe

#include <iostream>
#include <vector>
#include <omp.h>

#include <climits>
#include <cstdlib>

using namespace std;

int main() {
    int n = 10000000; // 10 million elements
    vector<int> arr(n);
    
    // Generate random data
    for(int i = 0; i < n; i++) {
        arr[i] = rand() % 10000; 
    }

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    long long sum = 0; // long long prevents overflow on massive arrays

    cout << "--- SPPU LP5: Practical 3 (Array Size: " << n << ") ---\n";

    // 1. Parallel Min
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < n; i++) {
        if(arr[i] < min_val) min_val = arr[i];
    }
    cout << "Parallel Min: " << min_val << " \tTime: " << omp_get_wtime() - start << " sec\n";

    // 2. Parallel Max
    start = omp_get_wtime();
    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < n; i++) {
        if(arr[i] > max_val) max_val = arr[i];
    }
    cout << "Parallel Max: " << max_val << " \tTime: " << omp_get_wtime() - start << " sec\n";

    // 3. Parallel Sum
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++) {
        sum += arr[i];
    }
    cout << "Parallel Sum: " << sum << " \tTime: " << omp_get_wtime() - start << " sec\n";

    // 4. Parallel Average
    double average = (double)sum / n;
    cout << "Average     : " << average << "\n";

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