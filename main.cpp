#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <cstring>
#include <windows.h>

static constexpr int NUM_BUCKETS = 8;
int main()
{
    alignas(32) uint32_t pivot[8] = { 5, 10, 15, 20, 25, 30, 35, 40 };
    const size_t NUM_ELEMENTS = (1ULL * 1024 * 1024 * 1024) / sizeof(int);
    const size_t BUCKET_CAPACITY = static_cast<size_t>((NUM_ELEMENTS / NUM_BUCKETS) * 1.1);
    std::cout << "allocating " << NUM_ELEMENTS << " ints ("
        << (NUM_ELEMENTS * sizeof(int) / (1024.0 * 1024.0 * 1024.0))
        << " GB)...\n";
    std::cout << "bucket capacity: " << BUCKET_CAPACITY << " ints each\n";
    uint32_t* testNumbers = new uint32_t[NUM_ELEMENTS];
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 39);
    for (size_t i = 0; i < NUM_ELEMENTS; i++)
        testNumbers[i] = dist(rng);
    std::cout << "done and starting benchmark...\n";
    __m256i pivotVec = _mm256_load_si256((__m256i*)pivot);
    uint32_t* buckets[NUM_BUCKETS];
    uint32_t* bucketHeads[NUM_BUCKETS];
    for (int i = 0; i < NUM_BUCKETS; i++)
    {
        bucketHeads[i] = buckets[i] = new uint32_t[BUCKET_CAPACITY];
        memset(buckets[i], 0, BUCKET_CAPACITY * sizeof(uint32_t));
    }
    SetThreadAffinityMask(GetCurrentThread(), 1 << 4);
    uint32_t* endInput = testNumbers + NUM_ELEMENTS;
    // --- TIMED PASS---
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t* input = testNumbers; input < endInput; input++)
    {
        // prefetch input
        _mm_prefetch((char*)input + 2048, _MM_HINT_T2);
        uint32_t key = *input;
        //__m256i num = _mm256_set1_epi32(key);
        __m256i num = _mm256_castps_si256(_mm256_broadcast_ss((float*)input));
        __m256i result = _mm256_cmpgt_epi32(pivotVec, num);
        uint32_t bitmask = _mm256_movemask_ps(_mm256_castsi256_ps(result));
        uint64_t bucket = _tzcnt_u32(bitmask);
        uint32_t* p = bucketHeads[bucket];
        *p = key;
        bucketHeads[bucket] = p + 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    // --- PRINTING, CLEANUP ---
    double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    double million_keys_per_second = (NUM_ELEMENTS / elapsed_seconds) / 1e6;
    std::cout << "\n=== Benchmark Results TEST===\n";
    std::cout << "Elements processed : " << NUM_ELEMENTS << "\n";
    std::cout << "Time elapsed : " << elapsed_seconds << " seconds\n";
    std::cout << "Throughput : " << million_keys_per_second << " M keys/sec\n";
    std::cout << "\n=== Bucket Distribution ===\n";
    for (int i = 0; i < NUM_BUCKETS; i++)
    {
        std::cout << "Bucket " << i << ":"
            << " elements ("
            << (100.0 * (bucketHeads[i] - buckets[i]) / NUM_ELEMENTS) << "%)\n";
    }
    delete[] testNumbers;
    for (int i = 0; i < NUM_BUCKETS; i++)
        delete[] buckets[i];
}