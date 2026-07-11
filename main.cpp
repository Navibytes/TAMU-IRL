#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <cstring>


static constexpr int    NUM_BUCKETS  = 8;
int main()
{
    alignas(32) int pivot[8] = { 5, 10, 15, 20, 25, 30, 35, 40 };

    const size_t NUM_ELEMENTS = (1ULL * 1024 * 1024 * 1024) / sizeof(int);
    const size_t BUCKET_CAPACITY = static_cast<size_t>((NUM_ELEMENTS / NUM_BUCKETS) * 1.1);

    std::cout << "allocating " << NUM_ELEMENTS << " ints ("
        << (NUM_ELEMENTS * sizeof(int) / (1024.0 * 1024.0 * 1024.0))
        << " GB)...\n";
    std::cout << "bucket capacity: " << BUCKET_CAPACITY << " ints each\n";

    int* testNumbers = new int[NUM_ELEMENTS];
        
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 39);
    for (size_t i = 0; i < NUM_ELEMENTS; i++)
        testNumbers[i] = dist(rng);

    std::cout << "done and starting benchmark...\n";

    __m256i pivotVec = _mm256_load_si256((__m256i*)pivot);

    int* buckets[NUM_BUCKETS];
    for (int i = 0; i < NUM_BUCKETS; i++)
        buckets[i] = new int[BUCKET_CAPACITY];
    size_t bucketHeads[NUM_BUCKETS] = {};
    for (int i = 0; i < NUM_BUCKETS; i++)
        memset(buckets[i], 0, BUCKET_CAPACITY * sizeof(int));

    volatile int64_t sink = 0;
    // --- TIMED PASS---
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_ELEMENTS; i++)
    {
        __m256i num          = _mm256_set1_epi32(testNumbers[i]);
        __m256i result       = _mm256_cmpgt_epi32(pivotVec, num);
        int     bitmask      = _mm256_movemask_epi8(result);
        int     trailingZeros = _tzcnt_u32(bitmask);
        int     bucket       = trailingZeros / 4;
        
        size_t  pos = bucketHeads[bucket]++;
        buckets[bucket][pos]  = testNumbers[i]; 
        
        // Prefetch the next cache line of the destination, branchlessly
        /*
        _mm_prefetch(
            reinterpret_cast<const char*>(&buckets[bucket][pos + 16]),
            _MM_HINT_T1
        );
        */
        
    }

    auto end = std::chrono::high_resolution_clock::now();

        

    // --- PRINTING, CLEANUP ---
    double elapsed_seconds       = std::chrono::duration<double>(end - start).count();
    double million_keys_per_second = (NUM_ELEMENTS / elapsed_seconds) / 1e6;

    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << "Elements processed : " << NUM_ELEMENTS << "\n";
    std::cout << "Time elapsed       : " << elapsed_seconds << " seconds\n";
    std::cout << "Throughput         : " << million_keys_per_second << " M keys/sec\n";

    std::cout << "\n=== Bucket Distribution ===\n";
    for (int i = 0; i < NUM_BUCKETS; i++)
    {
        std::cout << "Bucket " << i << ":"
            << " elements ("
            << (100.0 * bucketHeads[i] / NUM_ELEMENTS) << "%)\n";
    }

    delete[] testNumbers;
    for (int i = 0; i < NUM_BUCKETS; i++)
        delete[] buckets[i];
}