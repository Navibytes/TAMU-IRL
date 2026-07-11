#include <iostream>
#include <iterator>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <cstring>
#include <bitset>
//currently is 1600 Million keys per sec max, avg is 1500 M/s (on laptop vistual studio only)
alignas(32) static int left_count_table [256][8];
alignas(32) static int right_count_table [256][8];
static int left_amount [256];

void lookup_table(){
    for (int mask = 0; mask < 256; mask++){
        int indexL {};
        int indexR {};
        int left[8];
        int right[8];
        for (int i = 0; i<8; i++){
            if (mask & (1 << i)){
                right[indexR++] = i;
            }else{
                left[indexL++] = i;
            }
        }
        left_amount[mask] = indexL;
        for(int i = 0; i <8; i++){
            left_count_table[mask][i] = (i < indexL) ? left[i] : 0;
            right_count_table[mask][i] = (i < indexR) ? right[i] : 0; 
        }
    }
}

int main(){
    lookup_table();
    // initilzation of variables
    const int pivot {20};
    const size_t NUM_ELEMENTS = (1ULL*1024*1024*1024)/sizeof(int);
    int* keys = new int[NUM_ELEMENTS];
    int* left_Bucket = new int[NUM_ELEMENTS]{};
    int* right_Bucket = new int[NUM_ELEMENTS]{};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0,40);
    for (size_t i{}; i< NUM_ELEMENTS; i++){
        keys[i] = dist(rng);
    }
    int lheader {};
    int rheader {};
    const __m256i pivotVec = _mm256_set1_epi32(pivot); 
    //finished initialization 
    auto start =  std::chrono::high_resolution_clock::now();
    for(size_t i {}; i < NUM_ELEMENTS; i+=8){
        __m256i keysVec = _mm256_load_si256((__m256i*) &keys[i]);
        __m256i cmp = _mm256_cmpgt_epi32(keysVec, pivotVec);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        __m256i leftVecTable = _mm256_load_si256((__m256i*)left_count_table[mask]);
        __m256i  rightVecTable = _mm256_load_si256((__m256i*)right_count_table[mask]);
        __m256i left_vals = _mm256_permutevar8x32_epi32(keysVec, leftVecTable);
        __m256i right_vals = _mm256_permutevar8x32_epi32(keysVec,rightVecTable);
        _mm256_storeu_si256((__m256i*)&right_Bucket[rheader], right_vals);
        _mm256_storeu_si256((__m256i*)&left_Bucket[lheader], left_vals);
        lheader += left_amount[mask];
        rheader += (8-left_amount[mask]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();
    double mill_keys_sec = (NUM_ELEMENTS/ seconds) / 1e6;

    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << "Elements processed : " << NUM_ELEMENTS << "\n";
    std::cout << "Time elapsed       : " << seconds << " seconds\n";
    std::cout << "Throughput         : " << mill_keys_sec << " M keys/sec\n";

    delete[] keys;
    delete[] left_Bucket;
    delete[] right_Bucket;
}