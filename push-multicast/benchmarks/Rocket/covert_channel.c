// Required for setting CPU affinity - MUST BE THE FIRST LINE
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <x86intrin.h>
#include <sched.h>

// --- Configuration ---
#define SHARED_MEM_SIZE 4096    // Size of the shared memory region
#define FLAG_MEM_SIZE 4096      // Size of the synchronization flag region
#define MAX_MSG_SIZE 256        // Maximum message size
#define SECRET_MESSAGE "This is a secret message!" // The message to be sent

// Helper macro for histogram calibration
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// --- gem5-Friendly Synchronization ---
// Spin-wait (busy-loop) to create delays in gem5 instead of usleep/sched_yield
void spin_wait(volatile long count) {
    for (volatile long i = 0; i < count; i++);
}

// --- Cache Utility Functions ---

// Accesses a memory address using inline assembly.
static inline void maccess(void* p) {
    asm volatile ("movq (%0), %%rax\n"
        :
        : "c" (p)
        : "rax");
}

// *** MODIFICATION: Flushes a cache line using inline assembly. ***
static inline void flush(void* p) {
    asm volatile ("clflush 0(%0)\n"
        :
        : "c" (p)
        : "rax");
}

// Measures the time taken to access a memory address using rdtscp.
static inline uint64_t measure_access_time(void* addr) {
    uint64_t start, end;
    unsigned int junk; // To hold the CPU ID from rdtscp

    _mm_mfence(); // Ensure memory operations complete before timing
    start = __rdtscp(&junk);
    maccess(addr);
    _mm_mfence(); // Ensure maccess completes before timing
    end = __rdtscp(&junk);

    return end - start;
}

// --- Shared Data Structure ---
typedef struct {
    volatile char* shared_data;     // For data transmission
    volatile char* sync_flags;      // For synchronization
    pthread_barrier_t barrier;
    unsigned int threshold;
} channel_t;

// --- Receiver (Spy) Thread ---

// Calibrates the cache hit/miss threshold using a histogram method.
unsigned int calibrate_threshold(void* mem) {
    printf("[Calibrating] Determining cache hit/miss threshold using histograms...\n");
    int sample_size = 200; // Number of bins in the histogram
    size_t hit_histogram[sample_size];
    size_t miss_histogram[sample_size];
    memset(hit_histogram, 0, sizeof(hit_histogram));
    memset(miss_histogram, 0, sizeof(miss_histogram));

    // 1. Measure cache hit latencies
    maccess(mem); // Bring into cache
    for (int i = 0; i < 4*1024*128; i++) {
        size_t d = measure_access_time(mem);
        if (i<10) printf("hit time: %zu\n", d);
        hit_histogram[MIN(sample_size - 1, d / 5)]++; // Bin the results
    }

    // 2. Measure cache miss latencies
    for (int i = 0; i <4*1024*1024; i++) {
        // *** MODIFICATION: Use inline assembly flush function ***
        flush(mem);
        _mm_mfence();
        size_t d = measure_access_time(mem);
                if (i<10) printf("miss time: %zu\n", d);
        miss_histogram[MIN(sample_size - 1, d / 5)]++; // Bin the results
    }

    // 3. Find the peaks for hit and miss times
    size_t hit_max = 0;
    size_t hit_max_i = 0;
    size_t miss_min_i = 0;

    for (int i = 0; i < sample_size; i++) {
        if (hit_max < hit_histogram[i]) {
            hit_max = hit_histogram[i];
            hit_max_i = i;
        }
        if (miss_histogram[i] > 10 && miss_min_i == 0) { // Find first significant miss bin
            miss_min_i = i;
        }
    }

    // 4. Find the "valley" between the two peaks
    size_t min_overlap = -1UL;
    size_t min_overlap_i = 0;

    // Search for the minimum overlap point between the hit peak and miss start
    for (size_t i = hit_max_i; i < miss_min_i; i++) {
        if (min_overlap > (hit_histogram[i] + miss_histogram[i])) {
            min_overlap = hit_histogram[i] + miss_histogram[i];
            min_overlap_i = i;
        }
    }
    printf("[Calibrating] Hit peak at ~%lu cycles, Miss peak starts at ~%lu cycles.\n", hit_max_i * 5, miss_min_i * 5);
    // Sanity check
    if (miss_min_i <= hit_max_i || min_overlap_i == 0) {
        printf("[Warning] Calibration failed to find a clear threshold. Defaulting to 150.\n");
        return 150;
    }

    unsigned int threshold = min_overlap_i * 5;
    printf("[Calibrating] Hit peak at ~%lu cycles, Miss peak starts at ~%lu cycles.\n", hit_max_i * 5, miss_min_i * 5);
    printf("[Calibrating] Determined Threshold: %u cycles.\n", threshold);

    return threshold;
}

void* receiver_thread(void* arg) {
    // Pin this thread to CPU 1
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np receiver");
    }
    printf("[Receiver] Pinned to CPU 1.\n");

    channel_t* channel = (channel_t*)arg;
    char received_msg[MAX_MSG_SIZE] = {0};
    int msg_index = 0;

    // --- 1. Calibrate ---
    channel->threshold = calibrate_threshold((void*)channel->shared_data);
    if (channel->threshold == 0) {
        fprintf(stderr, "Calibration failed. Exiting receiver.\n");
        return NULL;
    }

    pthread_barrier_wait(&channel->barrier);

    // --- 2. Synchronization Handshake ---
    printf("[Receiver] Waiting for synchronization pattern (1010101011111111)...\n");
    int sync_pattern = 0;

    for (int b = 15; b >= 0; b--) {
        // Signal ready for the next bit
        channel->sync_flags[0] = 1;

        // Wait for sender to send the bit
        while(channel->sync_flags[1] == 0) { spin_wait(100); }
        channel->sync_flags[1] = 0; // Clear the flag

        // Read the bit
        uint64_t time = measure_access_time((void*)channel->shared_data);
        int bit = (time < channel->threshold) ? 1 : 0;
        sync_pattern = ((sync_pattern << 1) | bit) & 0xFFFF;

        printf("\r[Receiver Debug] Received bit: %d, Current Pattern: 0x%04X", bit, sync_pattern);
        fflush(stdout);
    }
    printf("\n");

    if (sync_pattern == 0xAAFF) {
        channel->sync_flags[2] = 1; // Signal that sync was successful
        printf("[Receiver] Sync pattern detected!\n");
    } else {
        printf("[Receiver] Sync pattern FAILED. Exiting.\n");
        return NULL;
    }

    // --- 3. Receive Data ---
    for (size_t i = 0; i < strlen(SECRET_MESSAGE); i++) {
        char current_char = 0;
        for (int j = 7; j >= 0; j--) {
            while(channel->sync_flags[128] == 0) { spin_wait(100); }
            channel->sync_flags[128] = 0;

            uint64_t time = measure_access_time((void*)channel->shared_data);
            int bit = (time < channel->threshold) ? 1 : 0;
            current_char |= (bit << j);

            channel->sync_flags[256] = 1;
        }
        received_msg[msg_index++] = current_char;
        printf("\r[Receiver] Receiving... [ %s ]", received_msg);
        fflush(stdout);
    }

    printf("\n[Receiver] Finished.\n");
    printf("[Receiver] Full Message Received: \"%s\"\n", received_msg);

    return NULL;
}

// --- Sender (Trojan) ---

// Transmits a single bit (1 for access, 0 for flush).
void transmit_bit(int bit, volatile char* addr) {
    if (bit) {
        maccess((void*)addr);
    } else {
        // *** MODIFICATION: Use inline assembly flush function ***
        flush((void*)addr);
    }
    _mm_mfence();
}

void run_sender(void* arg) {
    // Pin this thread (main) to CPU 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np sender");
    }
    printf("[Sender] Pinned to CPU 0.\n");

    channel_t* channel = (channel_t*)arg;
    const char* message = SECRET_MESSAGE;

    pthread_barrier_wait(&channel->barrier);

    // --- 2. Synchronization Handshake ---
    printf("[Sender] Sending synchronization pattern...\n");

    for (int b = 15; b >= 0; b--) {
        // Wait for the receiver to be ready for the next bit
        while(channel->sync_flags[0] == 0) { spin_wait(100); }
        channel->sync_flags[0] = 0; // Clear the flag

        // Send the bit
        int bit_to_send = (0xAAFF >> b) & 1;
        transmit_bit(bit_to_send, channel->shared_data);

        // Signal that the bit has been sent
        channel->sync_flags[1] = 1;
    }

    // Wait for receiver to confirm the full pattern
    while(channel->sync_flags[2] == 0) { spin_wait(100); }
    printf("[Sender] Sync confirmed by receiver. Starting message transmission.\n");


    // --- 3. Send Data ---
    for (size_t i = 0; i < strlen(message); i++) {
        char current_char = message[i];
        for (int j = 7; j >= 0; j--) {
            int bit = (current_char >> j) & 1;

            channel->sync_flags[128] = 1;
            transmit_bit(bit, channel->shared_data);

            while(channel->sync_flags[256] == 0) { spin_wait(100); }
            channel->sync_flags[256] = 0;
        }
    }

    printf("[Sender] Finished sending message.\n");
}

// --- Main Thread ---
int main() {
    pthread_t receiver_tid;
    channel_t channel;

    // 1. Allocate memory
    channel.shared_data = (volatile char*)malloc(SHARED_MEM_SIZE);
    channel.sync_flags = (volatile char*)malloc(FLAG_MEM_SIZE);
    if (!channel.shared_data || !channel.sync_flags) {
        perror("malloc");
        return 1;
    }
    memset((void*)channel.shared_data, 0, SHARED_MEM_SIZE);
    memset((void*)channel.sync_flags, 0, FLAG_MEM_SIZE);

    // 2. Initialize barrier for 2 threads
    pthread_barrier_init(&channel.barrier, NULL, 2);

    // 3. Create ONLY the receiver thread
    printf("Creating receiver thread...\n");
    if (pthread_create(&receiver_tid, NULL, receiver_thread, &channel) != 0) {
        perror("pthread_create receiver");
        return 1;
    }

    // Main thread BECOMES the sender
    run_sender(&channel);

    // 4. Wait for the receiver thread to finish
    pthread_join(receiver_tid, NULL);

    // 5. Clean up
    pthread_barrier_destroy(&channel.barrier);
    free((void*)channel.shared_data);
    free((void*)channel.sync_flags);

    printf("\nCovert channel simulation finished.\n");
    return 0;
}
