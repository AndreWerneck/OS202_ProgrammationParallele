#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <mpi.h>

#define N 1000

using namespace std;

int m_rank, n_ranks;

// C++ program for Merge Sort
#include <iostream>
using namespace std;

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
void merge(int array[], int const left, int const mid,
           int const right)
{
    auto const subArrayOne = mid - left + 1;
    auto const subArrayTwo = right - mid;

    // Create temp arrays
    auto *leftArray = new int[subArrayOne],
         *rightArray = new int[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    auto indexOfSubArrayOne = 0,   // Initial index of first sub-array
        indexOfSubArrayTwo = 0;    // Initial index of second sub-array
    int indexOfMergedArray = left; // Initial index of merged array

    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
    {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo])
        {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else
        {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne)
    {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo)
    {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;
}

// begin is for left index and end is
// right index of the sub-array
// of arr to be sorted */
void mergeSort(int array[], int const begin, int const end)
{
    if (begin >= end)
        return; // Returns recursively

    auto mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
}

void init_random(int *array)
{
    for (int i = 0; i < N; i++)
    {
        array[i] = rand() % N;
    }
}

void fill_buckets(int *array, int buckets[][N], int buckets_size[])
{
    for (int i = m_rank; i < N; i += n_ranks)
    {
        int b = array[i] * n_ranks / N;
        int i_b = buckets_size[b];
        buckets[b][i_b] = array[i];
        buckets_size[b] += 1;
    }
}

void fill_zero(int *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        array[i] = 0;
    }
}

int sum(int *array, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += array[i];
    }
    return sum;
}

void calc_displacements(int *array, int *dis, int n)
{
    dis[0] = 0;
    for (int i = 1; i < n; i++)
    {
        dis[i] = dis[i - 1] + array[i - 1];
    }
}

void print(int *array, int n)
{   
    cout << "Array: ";
    for (int i = 0; i < n; i++)
    {
        cout << array[i];
        if (i < n - 1)
            cout << ",";
    }
    cout << endl;
}

int main(int nargs, char *argv[])
{
    srand(0);

    int array[N];

    MPI_Init(&nargs, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    init_random(array);

    int buckets[n_ranks][N];
    int buckets_size[n_ranks];

    int bucket_rank[N];
    int bucket_rank_sizes[n_ranks];
    int displacements[n_ranks];
    int bucket_rank_size, total_size;

    fill_zero(buckets_size, n_ranks);
    fill_zero(displacements, n_ranks);
    fill_buckets(array, buckets, buckets_size);

    for (int b = 0; b < n_ranks; b++)
    {
        MPI_Gather(&buckets_size[b], 1, MPI_INT, bucket_rank_sizes, 1, MPI_INT, b, MPI_COMM_WORLD);
    }

    bucket_rank_size = sum(bucket_rank_sizes, n_ranks);
    calc_displacements(bucket_rank_sizes, displacements, n_ranks);

    for (int b = 0; b < n_ranks; b++)
    {
        MPI_Gatherv(buckets[b], buckets_size[b], MPI_INT, bucket_rank, bucket_rank_sizes, displacements, MPI_INT, b, MPI_COMM_WORLD);
    }

    // if (m_rank == 0) {
    //     MPI_Gatherv(buckets[0], buckets_size[0], MPI_INT, bucket_rank, bucket_rank_sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);
    // }
    // if (m_rank != 0) {
    //     MPI_Gatherv(buckets[0], buckets_size[0], MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    // }

    // cout << m_rank << " - My bucket of size " << bucket_rank_size << ": ";
    // for (int i = 0;i < bucket_rank_size;i++) {
    //     cout << bucket_rank[i] << ",";
    // }
    mergeSort(bucket_rank, 0, bucket_rank_size - 1);

    // cout << m_rank << " - My bucket of size " << bucket_rank_size << ": ";
    // cout << endl;

    fill_zero(bucket_rank_sizes, n_ranks);
    MPI_Gather(&bucket_rank_size, 1, MPI_INT, bucket_rank_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    calc_displacements(bucket_rank_sizes, displacements, n_ranks);
    MPI_Gatherv(bucket_rank, bucket_rank_size, MPI_INT, array, bucket_rank_sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    if (m_rank == 0) print(array, N);

    MPI_Finalize();

    return 0;
}