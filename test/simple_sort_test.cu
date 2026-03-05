#include <stdio.h>

__global__ void simple_sort_test(int *data, unsigned int valid_items) {
    // Use WarpMergeSort with ThreadsInWarp=2, ItemsPerThread=2
    // This should sort 4 items max, but we only have valid_items valid items
    
    // Simple test: manually implement what WarpMergeSort should do
    int tid = threadIdx.x;
    int lane_id = tid % 2;
    int warp_id = tid / 2;
    
    if (warp_id > 0) return;  // Only test first warp
    
    int thread_keys[2];
    int thread_offset = warp_id * 4 + lane_id * 2;
    
    // Load data
    for (int i = 0; i < 2; i++) {
        int local_idx = lane_id * 2 + i;
        if (local_idx < valid_items) {
            thread_keys[i] = data[thread_offset + i];
        } else {
            thread_keys[i] = 5;  // oob_default = ThreadsInWarp * ItemsPerThread + 1 = 5
        }
    }
    
    // Debug: print loaded keys
    if (lane_id == 0) {
        printf("Before sort: lane0=[%d,%d], lane1=[%d,%d]\n", 
               thread_keys[0], thread_keys[1], 
               data[warp_id * 4 + 2], data[warp_id * 4 + 3]);
    }
    __syncthreads();
    
    // Simple sort within thread (odd-even sort for 2 items)
    if (thread_keys[0] > thread_keys[1]) {
        int tmp = thread_keys[0];
        thread_keys[0] = thread_keys[1];
        thread_keys[1] = tmp;
    }
    
    // For ThreadsInWarp=2, we need to merge 2 threads
    // This is where CUB's merge sort would do its work
    // For now, let's just do a simple merge
    
    __syncthreads();
    
    // Debug: print after local sort
    printf("After local sort: lane%d=[%d,%d]\n", lane_id, thread_keys[0], thread_keys[1]);
    
    // Store result
    for (int i = 0; i < 2; i++) {
        int local_idx = lane_id * 2 + i;
        if (local_idx < valid_items) {
            data[thread_offset + i] = thread_keys[i];
        }
    }
}

int main() {
    int h_data[4] = {0, 0, 0, 0};  // Only first element is valid
    int *d_data;
    
    musaMalloc(&d_data, 4 * sizeof(int));
    musaMemcpy(d_data, h_data, 4 * sizeof(int), musaMemcpyHostToDevice);
    
    simple_sort_test<<<1, 2>>>(d_data, 1);
    musaDeviceSynchronize();
    
    musaMemcpy(h_data, d_data, 4 * sizeof(int), musaMemcpyDeviceToHost);
    
    printf("Result: [%d, %d, %d, %d]\n", h_data[0], h_data[1], h_data[2], h_data[3]);
    printf("Expected: [0, ?, ?, ?] (first element should be 0)\n");
    
    musaFree(d_data);
    return 0;
}
