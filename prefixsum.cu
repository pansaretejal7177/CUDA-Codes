// #include <stdio.h>

// void initWith(float val, float *arr, int N)
// {
//   for (int i = 0; i < N; i++)
//   {
//     arr[i] = val;
//   }
// }

// __global__ void prefixSum(float arr, float *res, float *ptemp, float ttemp, int N)
// {
//   int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//   int totalThreads = gridDim.x * blockDim.x;
//   int elementsPerThread = ceil(1.0 * N / totalThreads);

//   int start = threadId * elementsPerThread;
//   int count = 0;
//   float *sums = new float[elementsPerThread];
//   float sum = 0;

//   for (int i = start; i < N && count < elementsPerThread; i++, count++) {
//     sum += arr[i];
//     sums[count] = sum;
//   }

//   float localSum;
//   if (count)
//     localSum = sums[count - 1];
//   else
//     localSum = 0;
//   ptemp[threadId] = localSum;
//   ttemp[threadId] = localSum;

//   __syncthreads();

//   if (totalThreads == 1) {
//     for (int i = 0; i < N; i++)
//       res[i] = sums[i];
//   } else {
//     int d = 0; // log2(totalThreads)
//     int x = totalThreads;
//     while (x > 1) {
//       d++;
//       x = x >> 1;
//     }

//     x = 1;
//     for (int i = 0; i < 2*d; i++) {
//       int tsum = ttemp[threadId];

//       __syncthreads();

//       int newId = threadId / x;
//       if (newId % 2 == 0) {
//         int nextId = threadId + x;
//         ptemp[nextId] += tsum;
//         ttemp[nextId] += tsum;
//       } else {
//         int nextId = threadId - x;
//         ttemp[nextId] += tsum;
//       }

//       x = x << 1;
//     }

//     __syncthreads();

//     float diff = ptemp[threadId] - localSum;
//     for (int i = start, j = 0; i < N && j < count; i++, j++) {
//       res[i] = sums[j] + diff;
//     }
//   }
// }

// void checkRes(float arr, float *res, int N, float *ptemp, float ttemp)
// {
//   float sum = 0;
//   for (int i = 0; i < N; i++)
//   {
//     sum += arr[i];
//     if (sum != res[i])
//     {
//       printf("FAIL: res[%d] - %0.0f does not equal %0.0f\n", i, res[i], sum);
//       exit(1);
//     }
//   }
//   printf("SUCCESS! All prefix sums added correctly.\n");
// }

// int main()
// {
//   const int N = 1000000;
//   size_t size = N * sizeof(float);

//   float *arr;
//   float *res;

//   cudaMallocManaged(&arr, size);
//   cudaMallocManaged(&res, size);

//   initWith(2, arr, N);
//   initWith(0, res, N);

//   int blocks = 1;
//   int threadsPerBlock = 32;
//   int totalThreads = blocks * threadsPerBlock;

//   float *ptemp;
//   float *ttemp;
//   cudaMallocManaged(&ptemp, totalThreads * sizeof(float));
//   cudaMallocManaged(&ttemp, totalThreads * sizeof(float));

//   prefixSum<<<blocks, threadsPerBlock>>>(arr, res, ptemp, ttemp, N);
//   cudaDeviceSynchronize();

//   checkRes(arr, res, N, ptemp, ttemp);

//   cudaFree(arr);
//   cudaFree(res);
//   cudaFree(ttemp);
//   cudaFree(ptemp);
// }

#include <bits/stdc++.h>

using std::accumulate;
using std::cout;
using std::generate;
using std::vector;

#define SHMEM_SIZE 256

void init(vector<int> &h_v)
{
    for (int i = 0; i < h_v.size(); i++)
        h_v[i] = rand() % 10;
}

__global__ void prefixSum(int *v, int *v_r)
{
    __shared__ int partial_sum[SHMEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main()
{
    int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    vector<int> h_v(N);
    vector<int> h_v_r(N);

    init(h_v);
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

    const int TB_SIZE = 256;

    int GRID_SIZE = N / TB_SIZE;

    prefixSum<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

    prefixSum<<<1, TB_SIZE>>>(d_v_r, d_v_r);

    cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

    cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}