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

root@labpc:/AE-Folder/push-multicast/benchmarks/Rocket# b
/AE-Folder/push-multicast/benchmarks
root@labpc:/AE-Folder/push-multicast/benchmarks# cat M
cat: M: No such file or directory
root@labpc:/AE-Folder/push-multicast/benchmarks# cat Makefile
BENCH_TOP=$(shell pwd)

GEM_FORGE_BENCH=omp_conv3d2 omp_conv3d2_no_unroll omp_conv3d2_unroll \
                                omp_conv3d2_unroll_xy \
                                omp_dense_mv omp_dense_mv_blk omp_dense_mv_out \
                                pathfinder_kernel

BENCHDIR=${GEM_FORGE_TOP}/transform/benchmark/GemForgeMicroSuite
LLVM_CFLAGS=-O3 -DGEM_FORGE -mavx512f -fopenmp -std=c11 -gline-tables-only

LLVM_RELEASE=${GEM_FORGE_TOP}/llvm/install-release/bin
LLVM_CC=${LLVM_RELEASE}/clang

GEM_FORGE_TRANSFORM_SO=${GEM_FORGE_TOP}/transform/build/src/libLLVMTDGPass.so

# We use debug version to enable the debug flags.
LLVM_DEBUG=${GEM_FORGE_TOP}/llvm/install-debug/bin

GF_GEM5_INC=${GEM_FORGE_TOP}/gem5/include
GF_GEM5_OPS=${GEM_FORGE_TOP}/gem5/util/m5/m5op_x86.S

GEM_FORGE_BIN_DIR=${BENCH_TOP}/bin/gem-forge

GEM5_INC=${BENCH_TOP}/../gem5/include
GEM5_OPS=${BENCH_TOP}/../gem5/util/m5/src/abi/x86/m5op.S

CACHEBW_DIR=${BENCH_TOP}/ArchBenchSuite/level0/cachebw

READBW_MULTILEVEL_DIR=${BENCH_TOP}/ArchBenchSuite/level0/readbw_multilevel


.PHONY: all clean cachebw rodinia gem-forge parsec $(GEM_FORGE_BENCH) $(RODINIA_OMP_DIRS)

all: cachebw readbw_multilevel mlp omp_conv3d2_unroll_xy.exe omp_dense_mv_blk_256kB.exe omp_dense_mv_blk_512kB.exe omp_dense_mv_blk_1024kB.exe rodinia parsec

mlp:
        mkdir -p ${BENCH_TOP}/bin
        $(MAKE) SSE=3 -C libxsmm
        gcc -DNDEBUG -DLIBXSMM_TARGET_ARCH=1006 -DLIBXSMM_OPENMP_SIMD -I./libxsmm/samples/deeplearning/mlpdriver -I./libxsmm/include -fPIC --static -g -Wall -O2 -fopenmp -fopenmp-simd -funroll-loops -ftree-vectorize -fdata-sections -ffunction-sections -fvisibility=hidden -pthread -msse3 -DGEM5 -I${GEM5_INC} -c libxsmm/samples/deeplearning/mlpdriver/mlp_example_f32.c -o mlp.o
        gcc -static -o ${BENCH_TOP}/bin/mlp.exe mlp.o ${GEM5_OPS} -I${GEM5_INC} -L${BENCH_TOP}/libxsmm/lib/ -Wl,--as-needed ${BENCH_TOP}/libxsmm/lib/libxsmmext.a -Wl,--no-as-needed  ${BENCH_TOP}/libxsmm/lib/libxsmm.a -Wl,--as-needed ${BENCH_TOP}/libxsmm/lib/libxsmmnoblas.a -Wl,--no-as-needed -Wl,--gc-sections -Wl,-z,relro,-z,now -Wl,--as-needed -lm -lrt -ldl -L/usr/lib/gcc/x86_64-linux-gnu/10/libgfortran.a -lquadmath -lm -Wl,--no-as-needed -fopenmp -Wl,--as-needed -lstdc++ -Wl,--no-as-needed -pthread
        rm mlp.o

cachebw:
        mkdir -p ${BENCH_TOP}/bin
        # gcc -g -O2 -fopenmp -fstrict-aliasing -static -mavx2 -DGEM5 ${CACHEBW_DIR}/../common/perf_counter_markers.c ${CACHEBW_DIR}/cachebw.c -o cachebw.exe
        gcc -g -O2 -fopenmp -fstrict-aliasing -static -mavx2 -DGEM5 -I${GEM5_INC} ${GEM5_OPS} ${CACHEBW_DIR}/../common/perf_counter_markers.c ${CACHEBW_DIR}/cachebw.c -o cachebw.exe
        mv cachebw.exe ${BENCH_TOP}/bin/

pure-cachebw:
        mkdir -p ${BENCH_TOP}/bin
        gcc -O2 -fopenmp -fstrict-aliasing -static -mavx2 ${CACHEBW_DIR}/../common/counters.c ${CACHEBW_DIR}/cachebw.c -o cachebw.exe
        mv cachebw.exe ${BENCH_TOP}/bin/pure-cachebw.exe

readbw_multilevel:
        mkdir -p ${BENCH_TOP}/bin
        gcc -g -O2 -fopenmp -static -g -mavx2 -fstrict-aliasing -DGEM5 -I${GEM5_INC} ${GEM5_OPS} ${READBW_MULTILEVEL_DIR}/readbw_multilevel.c ${READBW_MULTILEVEL_DIR}/../common/perf_counter_markers.c -o readbw_multilevel.exe
        mv readbw_multilevel.exe ${BENCH_TOP}/bin/

gem-forge: ${GEM_FORGE_BIN_DIR} $(GEM_FORGE_BENCH)

${GEM_FORGE_BIN_DIR}:
        mkdir -p $@

$(GEM_FORGE_BENCH):
        ${LLVM_CC} -static -o ${GEM_FORGE_BIN_DIR}/$@.exe ${BENCHDIR}/$@/$@.c ${LLVM_CFLAGS} -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}

omp_conv3d2_unroll_xy.exe: ${BENCHDIR}/omp_conv3d2_unroll_xy/omp_conv3d2_unroll_xy.c
        mkdir -p ${GEM_FORGE_BIN_DIR}
        ${LLVM_CC} -g -static -o $@ $^ ${LLVM_CFLAGS} -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}
        mv omp_conv3d2_unroll_xy.exe ${GEM_FORGE_BIN_DIR}

clean_omp_conv3d2_unroll_xy:
        rm ${GEM_FORGE_BIN_DIR}/omp_conv3d2_unroll_xy.exe

omp_dense_mv_blk_256kB.exe: ${BENCHDIR}/omp_dense_mv_blk/omp_dense_mv_blk.c
        mkdir -p ${GEM_FORGE_BIN_DIR}
        ${LLVM_CC} -static -o $@ $^ ${LLVM_CFLAGS} -DL2Cache_256kB -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}
        mv omp_dense_mv_blk_256kB.exe ${GEM_FORGE_BIN_DIR}

omp_dense_mv_blk_512kB.exe: ${BENCHDIR}/omp_dense_mv_blk/omp_dense_mv_blk.c
        mkdir -p ${GEM_FORGE_BIN_DIR}
        ${LLVM_CC} -static -o $@ $^ ${LLVM_CFLAGS} -DL2Cache_512kB -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}
        mv omp_dense_mv_blk_512kB.exe ${GEM_FORGE_BIN_DIR}

omp_dense_mv_blk_1024kB.exe: ${BENCHDIR}/omp_dense_mv_blk/omp_dense_mv_blk.c
        mkdir -p ${GEM_FORGE_BIN_DIR}
        ${LLVM_CC} -static -o $@ $^ ${LLVM_CFLAGS} -DL2Cache_1024kB -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}
        mv omp_dense_mv_blk_1024kB.exe ${GEM_FORGE_BIN_DIR}

pathfinder.exe: ${BENCHDIR}/pathfinder_kernel/pathfinder_kernel.c
        mkdir -p ${GEM_FORGE_BIN_DIR}
        ${LLVM_CC} -static -o $@ $^ ${LLVM_CFLAGS} -lomp -lpthread -Wl,--no-as-needed -ffp-contract=off -ldl -I${GF_GEM5_INC} ${GF_GEM5_OPS}
        mv pathfinder.exe ${GEM_FORGE_BIN_DIR}

clean_pathfinder:
        rm ${GEM_FORGE_BIN_DIR}/pathfinder.exe

clean-gem-forge:
        rm -rf ${GEM_FORGE_BIN_DIR}


RODINIA_BASE_DIR=$(BENCH_TOP)/rodinia

RODINIA_BIN_DIR=$(BENCH_TOP)/bin/rodinia

#RODINIA_OMP_DIRS := b+tree backprop bfs cfd heartwall hotspot kmeans lavaMD leukocyte lud nn nw srad streamcluster particlefilter pathfinder mummergpu
RODINIA_OMP_DIRS := backprop bfs lud particlefilter pathfinder-avx512

rodinia:
        mkdir -p $(RODINIA_BIN_DIR)
        cd $(RODINIA_BASE_DIR)/openmp/backprop;             $(MAKE) -f Makefile.gem5; cp backprop.exe $(RODINIA_BIN_DIR)
        cd $(RODINIA_BASE_DIR)/openmp/bfs;                  $(MAKE) -f Makefile.gem5; cp bfs.exe $(RODINIA_BIN_DIR)
        cd $(RODINIA_BASE_DIR)/openmp/lud;                  $(MAKE) -f Makefile.gem5; cp lud.exe $(RODINIA_BIN_DIR)
        cd $(RODINIA_BASE_DIR)/openmp/particlefilter;       $(MAKE) -f Makefile.gem5; cp particlefilter.exe $(RODINIA_BIN_DIR)
        cd $(RODINIA_BASE_DIR)/openmp/pathfinder-avx512;    $(MAKE) -f Makefile.gem5; cp pathfinder.exe $(RODINIA_BIN_DIR)


PARSECDIR=$(BENCH_TOP)/parsec-3.0
PARSEC_BIN_DIR=$(BENCH_TOP)/bin/parsec
PARSEC_APPS=blackscholes bodytrack fluidanimate freqmine swaptions
PARSEC_KERNELS=`ls $(PARSECDIR)/pkgs/kernels`

parsec:
        mkdir -p $(PARSEC_BIN_DIR)
        cd $(PARSECDIR); ./bin/parsecmgmt -a build -p parsecapps -c gcc-gem5hooks
        for bench in $(PARSEC_APPS); do mkdir $(PARSEC_BIN_DIR)/$$bench; cp $(PARSECDIR)/pkgs/apps/$$bench/inst/*.gcc-gem5hooks/* -rf $(PARSEC_BIN_DIR)/$$bench/; done

# for timing channel attack by Rocket
CUSTOM_BIN_DIR=$(BENCH_TOP)/bin/Rocket
GEM5_GCC = /usr/bin/gcc
GEM5_CFLAGS = -O2 -static -pthread

attack:
        mkdir -p $(CUSTOM_BIN_DIR)
        $(GEM5_GCC) $(GEM5_CFLAGS) -o $(CUSTOM_BIN_DIR)/attack.exe $(BENCH_TOP)/Rocket/covert_channel.c

clean-attack:
        rm -rf $(CUSTOM_BIN_DIR)

clean-rodinia:
        rm -rf $(RODINIA_BIN_DIR)
        for dir in $(RODINIA_OMP_DIRS) ; do cd $(RODINIA_BASE_DIR)/openmp/$$dir; $(MAKE) clean -f Makefile.gem5; cd ../.. ; done

clean-cachebw:
        rm -f ${BENCH_TOP}/bin/cachebw.exe

clean-mlp:
        rm -f ${BENCH_TOP}/bin/mlp.exe

clean-parsec:
        rm -f $(PARSEC_BIN_DIR)

clean:
        rm -rf ${BENCH_TOP}/bin
root@labpc:/AE-Folder/push-multicast/benchmarks# b
/AE-Folder/push-multicast
root@labpc:/AE-Folder/push-multicast# cd utils/
root@labpc:/AE-Folder/push-multicast/utils# cat run-experiment.py
import os
import sys
import math
import argparse
import subprocess
import time
from copy import deepcopy
import multiprocessing as mp
import json


def calculate_closest_factors(num):
    '''Calculate the closest factors of an integer number'''

    col = int(math.sqrt(num))

    while num % col != 0:
        col -= 1

    row = num //col
    return row, col
# calculate_closest_factors() - end


def get_command(args, cmd, options):
    command = []

    command.append(args.gem5)
    command.append(f"--outdir={args.outdir}")

    # debug options
    if args.debug_file is not None:
        command.append(f"--debug-file={args.debug_file}")
    if args.debug_start is not None:
        command.append(f"--debug-start={args.debug_start}")
    if args.debug_end is not None:
        command.append(f"--debug-end={args.debug_end}")
    command.append(f"--debug-flags={args.debug_flags}")
    if args.sweep or args.launch_experiments or args.no_listener:
        command.append('--listener-mode=off')

    # runscript and system config
    if args.launch_experiments == "prepush-ack-bingo":
        command.append(f"./gem5/configs/example/se.py")
    else:
        command.append(f"./gem5/configs/example/se.py")


    command.append("--sys-clock=2GHz")

    # CPU options
    command.append(f"--num-cpus={args.num_cpus}")
    command.append(f"--cpu-type={args.cpu_type}")
    command.append("--cpu-clock=3.5GHz")

    # Memory hierarchy
    command.append("--caches")
    command.append("--l2cache")
    command.append("--ruby")
    command.append(f"--ruby-clock={args.ruby_clock}")
    command.append("--num-dirs=4")
    command.append(f"--num-l2caches={args.num_cpus}")
    command.append("--l0i_size=32kB")
    command.append("--l0i_assoc=8")
    command.append("--l0d_size=32kB")
    command.append("--l0d_assoc=8")
    command.append(f"--l1d_size={args.l2_size}")
    command.append("--l1d_assoc=16")
    command.append(f"--l2_size={args.llc_slice_size}")
    command.append("--l2_assoc=16")
    command.append("--mem-size=16GB")
    if args.gem5 == "./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt":
        if (args.launch_experiments == "sensitivity"):
            command.append(f"--feedback_threshold={args.feedback_threshold}")
            command.append(f"--allowed_window={args.allowed_window}")
        else:
            if args.num_cpus == 16:
                command.append("--feedback_threshold=16")
                command.append("--allowed_window=500")
            elif args.num_cpus == 64:
                command.append("--feedback_threshold=16")
                command.append("--allowed_window=1500")
            else:
                command.append(f"--feedback_threshold={args.feedback_threshold}")
                command.append(f"--allowed_window={args.allowed_window}")
    elif args.gem5 == "./gem5/build/X86_MESI_Three_Level_PrepushAck_Feedback_Restart_Ratio/gem5.opt":
        if (args.launch_experiments == "sensitivity"):
            command.append(f"--feedback_threshold={args.feedback_threshold}")
            command.append(f"--allowed_window={args.allowed_window}")
        else:
            if args.num_cpus == 16:
                command.append("--feedback_threshold=64")
                command.append("--allowed_window=500")
            elif args.num_cpus == 64:
                command.append("--feedback_threshold=8")
                command.append("--allowed_window=1500")
            else:
                command.append(f"--feedback_threshold={args.feedback_threshold}")
                command.append(f"--allowed_window={args.allowed_window}")

    # Prefetcher
    if args.enable_prefetch:
        command.append("--enable-prefetch")

    # NoC options
    command.append(f"--message-buffer-size={args.message_buffer_size}")
    command.append("--network=garnet")
    command.append("--router-latency=2")
    command.append("--link-latency=1")
    command.append(f"--link-width-bits={args.link_width_bits}")
    command.append("--topology=MeshDirCorners_XY")
    rows, _ = calculate_closest_factors(args.num_cpus)
    command.append(f"--mesh-rows={rows}")
    assert (64 * 8) % args.link_width_bits == 0 # 64: cacheline size in bytes
    buffers_per_data_vc = (64 * 8 // args.link_width_bits) + 1
    command.append(f"--buffers-per-data-vc={buffers_per_data_vc}")
    if args.enable_multicast:
        command.append("--enable-multicast")
        command.append("--asynchronous-multicast")
    command.append(f"--routing-algorithm={args.routing}")
    if args.hold_switch_for_multicast_only:
        command.append("--hold-switch-for-multicast-only")

    # Benchmark options
    command.append(f"--cmd={cmd}")
    command.append(f"--options=\"{options}\"")

    # Profiling and prepush options
    if args.profile_llc:
        command.append("--profile-llc-sharers")
        if args.window_cycles is not None:
            command.append(f"--window-cycles={args.window_cycles}")
        if args.profile_pc_low is not None:
            assert args.profile_pc_high is not None
            command.append(f"--profile-pc-low={args.profile_pc_low}")
            command.append(f"--profile-pc-high={args.profile_pc_high}")
        if args.profile_vaddr_low is not None:
            assert args.profile_vaddr_high is not None
            command.append(f"--profile-vaddr-low={args.profile_vaddr_low}")
            command.append(f"--profile-vaddr-high={args.profile_vaddr_high}")
    if args.always_prepush:
        command.append("--always-prepush")
    if args.prepush:
        command.append("--prepush")
        if args.profile_prepush:
            command.append("--profile-prepush")
        command.append(
                f"--noc-coherence-constraint={args.noc_coherence_constraint}")
        if args.prepush_filter:
            command.append("--prepush-filter")
        if args.prepush_filter_nodrop:
            command.append("--prepush-filter-nodrop")

    # Coalescing
    if args.coalescing:
        command.append("--coalescing")

    # Others
    command.append(f"--fast-forward={sys.maxsize}")
    if args.log:
        logfile_path = f"{args.outdir}/sim.log"
        logdir = os.path.dirname(logfile_path)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if args.sweep or args.launch_experiments:
            command.append(f"> {logfile_path} 2>&1")
        else:
            command.append(f"> {logfile_path} 2>&1 &")

    return command
# get_command() - end


def get_benchmark_cmd_options(args):
    benchmark = args.benchmark
    num_cpus = args.num_cpus
    test_input = args.test_input
    if args.scheme_name is None:
        scheme = ""
    else:
        scheme = args.scheme_name

    parsec_bin_dir = f"{os.getcwd()}/benchmarks/bin/parsec"
    assert os.path.exists(parsec_bin_dir)
    parsec_input_dir = f"{os.getcwd()}/benchmarks/inputs/parsec"
    assert os.path.exists(parsec_input_dir)
    if args.launch_experiments == "all-speedup":
        parsec_run_dir = f"{os.getcwd()}/benchmarks/runs/parsec"
        assert os.path.exists(parsec_run_dir)
    parsec_input = args.parsec_input

    rodinia_data = "./benchmarks/rodinia/data"

    if benchmark == "cachebw":
        cmd = "./benchmarks/bin/cachebw.exe"
        if test_input:
            options = f"65536 1 1 1 4 {num_cpus}" # 65536*8 = 512 KB
        else:
            options = f"1048576 1 1 1 1 {num_cpus}" # 1 iteration

    elif benchmark == "readbw_multilevel":
        cmd = "./benchmarks/bin/readbw_multilevel.exe"
        options = f"1024 4 4 {num_cpus} 4 1" #f"1024 4 4 {num_cpus} 4 1"

    elif benchmark == "mlp":
        cmd = "./benchmarks/bin/mlp.exe"
        if args.l2_size == "256kB":
            options = "5 256 4 F 8 8 8 1024 1024" #256kB
        elif args.l2_size == "512kB":
            options = "5 512 4 F 8 8 8 1024 1024" #512kB
        elif args.l2_size == "1MB":
            options = "5 1024 4 F 8 8 8 1024 1024" #1MB
        else:
            options = "5 256 4 F 8 8 8 1024 1024"

    elif benchmark == "conv3dfoowarm":
        cmd = "./benchmarks/bin/gem-forge/omp_conv3d2_unroll_xy.exe"
        options = f"{num_cpus}"

    elif benchmark == "mv":
        if args.l2_size == "256kB":
            cmd = "./benchmarks/bin/gem-forge/omp_dense_mv_blk_256kB.exe" #256kB
        elif args.l2_size == "512kB":
            cmd = "./benchmarks/bin/gem-forge/omp_dense_mv_blk_512kB.exe" #512kB
        elif args.l2_size == "1MB":
            cmd = "./benchmarks/bin/gem-forge/omp_dense_mv_blk_1024kB.exe" #1MB
        else:
            cmd = "./benchmarks/bin/gem-forge/omp_dense_mv_blk_256kB.exe"
        options = f"{num_cpus}"

    elif benchmark == "backprop":
        cmd = "./benchmarks/bin/rodinia/backprop.exe"
        if args.l2_size == "256kB":
            options = f"65535 {num_cpus}" #256kB
        elif args.l2_size == "512kB":
            options = f"131071 {num_cpus}" #512kB
        elif args.l2_size == "1MB":
            options = f"262143 {num_cpus}" #1MB
        else:
            options = f"65535 {num_cpus}"

    elif benchmark == "bfs":
        cmd = "./benchmarks/bin/rodinia/bfs.exe"
        if args.l2_size == "256kB":
            options = f"{num_cpus} {rodinia_data}/bfs/graph1MW_6.txt.data"
        elif args.l2_size == "512kB":
            options = f"{num_cpus} {rodinia_data}/bfs/graph4M.txt.data"
        elif args.l2_size == "1MB":
            options = f"{num_cpus} {rodinia_data}/bfs/graph4M.txt.data"
        else:
            options = f"{num_cpus} {rodinia_data}/bfs/graph1MW_6.txt.data"

    elif benchmark == "btree":
        cmd = "./benchmarks/bin/rodinia/btree.exe"
        options = f"cores {num_cpus} fake"

    elif benchmark == "cfd":
        cmd = "./benchmarks/bin/rodinia/euler3d_cpu_double.exe"
        iterations = args.cfd_iterations
        if iterations == 0:
            raise RuntimeError("--cfd-iterations is not specified!")
        if test_input:
            options = f"{rodinia_data}/cfd/fvcorr.domn.097K.data " + \
                    f"{num_cpus} {iterations}"
        else:
            options = f"{rodinia_data}/cfd/fvcorr.domn.193K.data " + \
                    f"{num_cpus} {iterations}"

    elif benchmark == "hotspot":
        cmd = "./benchmarks/bin/rodinia/hotspot.exe"
        options = f"1024 1024 8 {num_cpus} " + \
                "./benchmarks/rodinia/data/hotspot/temp_1024 " + \
                "./benchmarks/rodinia/data/hotspot/power_1024 " + \
                "./benchmarks/rodinia/data/hotspot/output_1024.out"

    elif benchmark == "hotspot3D":
        cmd = "./benchmarks/bin/rodinia/hotspot3D.exe"
        options = f"512 8 8 {num_cpus} " + \
                "./benchmarks/rodinia/data/hotspot3D/power_512x8 " + \
                "./benchmarks/rodinia/data/hotspot3D/temp_512x8 " + \
                "./benchmarks/rodinia/data/hotspot3D/output_512x8.out"

    elif benchmark == "kmeans":
       cmd = "./benchmarks/bin/rodinia/kmeans.exe"
       if test_input:
           infile = "./benchmarks/rodinia/data/kmeans/100.data"
       else:
           infile = "./benchmarks/rodinia/data/kmeans/kdd_cup.data"
       options = f"-i {infile} -b 1 -n {num_cpus}"

    elif benchmark == "lud":
        cmd = "./benchmarks/bin/rodinia/lud.exe"
        if test_input:
            options = f"-n {num_cpus} -s 128"
        else:
            if args.l2_size == "256kB":
                options = f"-n {num_cpus} -s 1024"
            elif args.l2_size == "512kB":
                options = f"-n {num_cpus} -s 2048"
            elif args.l2_size == "1MB":
                options = f"-n {num_cpus} -s 2048"
            else:
                options = f"-n {num_cpus} -s {args.lud_size}"

    elif benchmark == "nn":
        cmd = "./benchmarks/bin/rodinia/nn.exe"
        options = "./benchmarks/rodinia/data/nn/list750k.data.txt " + \
                f"5 30 90 {num_cpus}"

    elif benchmark == "nw":
        cmd = "./benchmarks/bin/rodinia/needle.exe"
        options = f"2048 10 {num_cpus}"

    elif benchmark == "particlefilter":
        cmd = "./benchmarks/bin/rodinia/particlefilter.exe"
        frames = args.particlefilter_frames
        if frames == 0:
            raise RuntimeError("--particlefilter-frames is not specified!")
        if test_input:
            options = f"-x 128 -y 128 -z {frames} -np 32768 -nt {num_cpus}"
        else:
            if args.l2_size == "256kB":
                options = f"-x 1000 -y 1000 -z {frames} -np 48000 -nt {num_cpus}"
            elif args.l2_size == "512kB":
                options = f"-x 1000 -y 1000 -z {frames} -np 192000 -nt {num_cpus}"
            elif args.l2_size == "1MB":
                options = f"-x 1000 -y 1000 -z {frames} -np 192000 -nt {num_cpus}"
            else:
                options = f"-x 1000 -y 1000 -z {frames} -np 48000 -nt {num_cpus}"

    elif benchmark == "pathfinder":
        cmd = "./benchmarks/bin/rodinia/pathfinder.exe"
        options = f"1500000 8 {num_cpus}"

    elif benchmark == "srad":
        cmd = "./benchmarks/bin/rodinia/srad.exe"
        options = f"512 2048 0 127 0 127 {num_cpus} 0.5 8"

    elif benchmark == "blackscholes":
        cmd = f"{parsec_bin_dir}/blackscholes/bin/blackscholes"
        options = f"{num_cpus} "
        if parsec_input == "test":
            options += f"{parsec_input_dir}/blackscholesin_4.txt " + \
                    f"{parsec_run_dir}/blackscholes/{scheme}_test_prices.txt"
        elif parsec_input == "small":
            options += f"{parsec_input_dir}/blackscholes/in_4K.txt " + \
                    f"{parsec_run_dir}/blackscholes/{scheme}_small_prices.txt"
        elif parsec_input == "medium":
            options += f"{parsec_input_dir}/blackscholes/in_16K.txt " + \
                    f"{parsec_run_dir}/blackscholes/{scheme}_medium_prices.txt"
        else:
            assert parsec_input == "large"
            options += f"{parsec_input_dir}/blackscholes/in_64K.txt " + \
                    f"{parsec_run_dir}/blackscholes/{scheme}_large_prices.txt"

    elif benchmark == "bodytrack":
        cmd = f"{parsec_bin_dir}/bodytrack/bin/bodytrack"
        if parsec_input == "test":
            options = f"{parsec_input_dir}/bodytrack/sequenceB_1 " + \
                    f"4 1 5 1 0 {num_cpus}"
        elif parsec_input == "small":
            options = f"{parsec_input_dir}/bodytrack/sequenceB_1 " + \
                    f"4 1 1000 5 0 {num_cpus}"
        elif parsec_input == "medium":
            options = f"{parsec_input_dir}/bodytrack/sequenceB_2 " + \
                    f"4 2 2000 5 0 {num_cpus}"
        else:
            assert parsec_input == "large"
            options = f"{parsec_input_dir}/bodytrack/sequenceB_4 " + \
                    f"4 4 4000 5 0 {num_cpus}"

    elif benchmark == "canneal":
        cmd = f"{parsec_bin_dir}/canneal/bin/canneal"
        if parsec_input == "test":
            options = f"{num_cpus} 5 100 {parsec_input_dir}/canneal/10.nets 1 {num_cpus}"
        elif parsec_input == "small":
            options = f"{num_cpus} 10000 2000 " + \
                    f"{parsec_input_dir}/canneal/100000.nets 32 {num_cpus}"
        elif parsec_input == "medium":
            options = f"{num_cpus} 15000 2000 " + \
                    f"{parsec_input_dir}/canneal/200000.nets 64 {num_cpus}"
        else:
            assert parsec_input == "large"
            options = f"{num_cpus} 15000 2000 " + \
                    f"{parsec_input_dir}/canneal/400000.nets 128 {num_cpus}"

    elif benchmark == "dedup":
        cmd = f"{parsec_bin_dir}/dedup/bin/dedup"
        if parsec_input == "test":
            options = f"-c -p -v -t {num_cpus} " + \
                    f"-i {parsec_input_dir}/dedup/test.dat " + \
                    f"-o {parsec_run_dir}/dedup/{scheme}_test_output.dat.ddp"
        elif parsec_input == "small":
            options = f"-c -p -v -t {num_cpus} " + \
                    f"-i {parsec_input_dir}/dedup/small_media.dat " + \
                    f"-o {parsec_run_dir}/dedup/{scheme}_small_output.dat.ddp"
        elif parsec_input == "medium":
            options = f"-c -p -v -t {num_cpus} " + \
                    f"-i {parsec_input_dir}/dedup/medium_media.dat " + \
                    f"-o {parsec_run_dir}/dedup/{scheme}_medium_output.dat.ddp"
        else:
            assert parsec_input == "large"
            options = f"-c -p -v -t {num_cpus} " + \
                    f"-i {parsec_input_dir}/dedup/large_media.dat " + \
                    f"-o {parsec_run_dir}/dedup/{scheme}_large_output.dat.ddp"

    elif benchmark == "fluidanimate":
        cmd = f"{parsec_bin_dir}/fluidanimate/bin/fluidanimate"
        if parsec_input == "test":
            options = f"{num_cpus} 1 " + \
                    f"{parsec_input_dir}/fluidanimate/in_5K.fluid " + \
                    f"{parsec_run_dir}/fluidanimate/{scheme}_test_out.fluid"
        elif parsec_input == "small":
            options = f"{num_cpus} 5 " + \
                    f"{parsec_input_dir}/fluidanimate/in_35K.fluid " + \
                    f"{parsec_run_dir}/fluidanimate/{scheme}_small_out.fluid"
        elif parsec_input == "medium":
            options = f"{num_cpus} 5 " + \
                    f"{parsec_input_dir}/fluidanimate/in_100K.fluid " + \
                    f"{parsec_run_dir}/fluidanimate/{scheme}_medium_out.fluid"
        else:
            assert parsec_input == "large"
            options = f"{num_cpus} 5 " + \
                    f"{parsec_input_dir}/fluidanimate/in_300K.fluid " + \
                    f"{parsec_run_dir}/fluidanimate/{scheme}_large_out.fluid"

    elif benchmark == "freqmine":
        cmd = f"{parsec_bin_dir}/freqmine/bin/freqmine"
        if parsec_input == "test":
            options = f"{parsec_input_dir}/freqmine/T10I4D100K_3.dat 1"
        elif parsec_input == "small":
            options = f"{parsec_input_dir}/freqmine/kosarak_250k.dat 220"
        elif parsec_input == "medium":
            options = f"{parsec_input_dir}/freqmine/kosarak_500k.dat 410"
        else:
            assert parsec_input == "large"
            options = f"{parsec_input_dir}/freqmine/kosarak_990k.dat 790"

    elif benchmark == "streamcluster":
        cmd = f"{parsec_bin_dir}/streamcluster/bin/streamcluster"
        if parsec_input == "test":
            options = f"2 5 1 10 10 5 none {parsec_run_dir}/streamcluster/{scheme}_test_output.txt {num_cpus}"
        elif parsec_input == "small":
            options = f"10 20 32 4096 4096 1000 none {parsec_run_dir}/streamcluster/{scheme}_small_output.txt {num_cpus}"
        else:
            assert parsec_input == "medium"
            options = f"10 20 64 8192 8192 1000 none {parsec_run_dir}/streamcluster/{scheme}_medium_output.txt {num_cpus}"

    elif benchmark == "swaptions":
        cmd = f"{parsec_bin_dir}/swaptions/bin/swaptions"
        if parsec_input == "test":
            options = f"-ns 1 -sm 5 -nt {num_cpus}"
        elif parsec_input == "small":
            options = f"-ns 16 -sm 10000 -nt {num_cpus}"
        elif parsec_input == "medium":
            options = f"-ns 32 -sm 20000 -nt {num_cpus}"
        else:
            assert parsec_input == "large"
            options = f"-ns 64 -sm 20000 -nt {num_cpus}"

    elif benchmark == "x264":
        cmd = f"{parsec_bin_dir}/x264/bin/x264"
        if parsec_input == "test":
            options = f"--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 " + \
                    f"--direct auto --b-pyramid --weightb --mixed-refs " + \
                    f"--no-fast-pskip --me umh --subme 7 " + \
                    f"--analyse b8x8,i4x4 --threads {num_cpus} " + \
                    f"-o {parsec_run_dir}/x264/{scheme}_test_eledream.264 " +\
                    f"{parsec_input_dir}/x264/eledream_32x18_1.y4m"
        elif parsec_input == "small":
            options = f"--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 " + \
                    f"--direct auto --b-pyramid --weightb --mixed-refs " + \
                    f"--no-fast-pskip --me umh --subme 7 " + \
                    f"--analyse b8x8,i4x4 --threads {num_cpus} -o " + \
                    f"{parsec_run_dir}/x264/{scheme}_small_eledream.264 " +\
                    f"{parsec_input_dir}/x264/eledream_640x360_32.y4m"
        else:
            assert parsec_input == "medium"
            options = f"--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 " + \
                    f"--direct auto --b-pyramid --weightb --mixed-refs " + \
                    f"--no-fast-pskip --me umh --subme 7 " + \
                    f"--analyse b8x8,i4x4 --threads {num_cpus} -o " + \
                    f"{parsec_run_dir}/x264/{scheme}_medium_eledream.264 " + \
                    f"{parsec_input_dir}/x264/eledream_640x360_8.y4m"


    elif benchmark == "covertchannel":
        # You may use any directory, adjust according to your artifact structure
        cmd = "benchmarks/bin/Rocket/attack.exe"
        options = ""  # no extra options needed, or set up as per your experiment


    else:
        # TODO: add other benchmarks
        raise NotImplementedError(f"{benchmark} not found")

    return cmd, options
# get_benchmark_cmd_options() - end


def run_gem5_instance(args):
    """ Run a simulation instance. """

    cmd, options = get_benchmark_cmd_options(args)

    command = get_command(args, cmd, options)
    command = ' '.join(command)

    if args.dry_run:
        print(command)
    else:
        start_time = time.time()
        print(f"Running '{command}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")

        returncode = subprocess.run(command, env=os.environ,
                shell=True).returncode

        end_time = time.time()
        print(f"Finished '{command}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}. Total time = {end_time - start_time}")

        if returncode != 0:
            raise RuntimeError(f"{command} -> {returncode}")
# run_gem5_instance() - end


def sweep(args):
    """ Sweep number of cpus and window cycles. """

    args_list = []

    for num_cpus in args.sweep_num_cpus:
        args.num_cpus = num_cpus

        for window_cycles in args.sweep_window_cycles:
            args.window_cycles = window_cycles

            args.outdir = f"m5out/{args.benchmark}-{num_cpus}cpus-{window_cycles}window"
            args.log = True

            args_list.append(deepcopy(args))

    if args.sweep_thread_pool_size is None:
        args.sweep_thread_pool_size = mp.cpu_count() // 2

    if len(args_list) < args.sweep_thread_pool_size:
        args.sweep_thread_pool_size = len(args_list)

    pool = mp.Pool(args.sweep_thread_pool_size)
    pool.map(run_gem5_instance, args_list)
    pool.close()
    pool.join()

    print("Complete all simulation jobs!")
# sweep() - end


def configure_experiments(args, scheme):
    """ Configure expeirments of all benchmarks for a scheme. """

    parsec_benchmarks = ["blackscholes", "bodytrack", "canneal", "dedup",
            "facesim", "ferret", "fluidanimate", "facesim", "freqmine", "raytrace",
            "streamcluster", "swaptions", "vips", "x264"]

    benchmarks = args.benchmarks
    if args.launch_experiments == "all-speedup":
        benchmarks = ["cachebw", "readbw_multilevel", "mlp", "conv3dfoowarm", "mv",
                    "backprop", "bfs", "lud", "particlefilter", "pathfinder", "blackscholes",
                    "bodytrack", "fluidanimate", "freqmine", "swaptions"]
    elif args.launch_experiments == "ablation":
        benchmarks = ["cachebw", "readbw_multilevel", "mlp", "conv3dfoowarm", "mv",
                    "backprop", "bfs", "lud", "particlefilter", "pathfinder"]
    elif args.launch_experiments == "link-study":
        benchmarks = ["cachebw", "readbw_multilevel", "mlp", "conv3dfoowarm", "mv",
                    "backprop", "bfs", "lud", "particlefilter", "pathfinder"]
    elif args.launch_experiments == "cache-size":
        benchmarks = ["cachebw", "readbw_multilevel", "mlp", "conv3dfoowarm", "mv",
                    "backprop", "bfs", "lud", "particlefilter", "pathfinder"]
    elif args.launch_experiments == "sensitivity":
        benchmarks = ["bfs", "conv3dfoowarm"]
    elif args.launch_experiments == "violin":
        benchmarks = ["mv"]
    else:
        benchmarks = benchmarks


    args_list = []

    if args.enable_prefetch:
        prefetch_postfix = "-prefetch"
    else:
        prefetch_postfix = ""

    num_cpus = args.num_cpus
    if (args.launch_experiments == "sensitivity"):
        m5outdir = f"m5out/{args.experiments_outdir}/{scheme}"
    else:
        m5outdir = f"m5out/{args.experiments_outdir}/{scheme}" + prefetch_postfix
    args.log = True

    for b, benchmark in enumerate(benchmarks):
        args.benchmark = benchmark

        if benchmark == "cfd":
            # use small input
            args.test_input = True
            args.cfd_iterations = 1
            args.outdir = f"{m5outdir}/{benchmark}-1iter-test-{num_cpus}cpus"
            args_list.append(deepcopy(args))

            args.cfd_iterations = 2
            args.outdir = f"{m5outdir}/{benchmark}-2iter-test-{num_cpus}cpus"
            args_list.append(deepcopy(args))

            # use large input
            args.test_input = False
            args.cfd_iterations = 1
            args.outdir = f"{m5outdir}/{benchmark}-1iter-{num_cpus}cpus"
            args_list.append(deepcopy(args))

            args.cfd_iterations = 2
            args.outdir = f"{m5outdir}/{benchmark}-2iter-{num_cpus}cpus"
            args_list.append(deepcopy(args))

        elif benchmark == "particlefilter":
            # use small input
            # args.test_input = True
            # args.particlefilter_frames = 1
            # args.outdir = f"{m5outdir}/{benchmark}-1fr-test-{num_cpus}cpus"
            # args_list.append(deepcopy(args))

            # args.particlefilter_frames = 2
            # args.outdir = f"{m5outdir}/{benchmark}-2fr-test-{num_cpus}cpus"
            # args_list.append(deepcopy(args))

            # # use large input
            # args.test_input = False
            # args.particlefilter_frames = 1
            # args.outdir = f"{m5outdir}/{benchmark}-1fr-{num_cpus}cpus"
            # args_list.append(deepcopy(args))

            args.particlefilter_frames = 2
            args.outdir = f"{m5outdir}/{benchmark}-2fr-{num_cpus}cpus"
            args_list.append(deepcopy(args))

        elif benchmark in parsec_benchmarks:
            if args.parsec_input_list:
                for parsec_input in args.parsec_input_list:
                    args.parsec_input = parsec_input
                    args.outdir = f"{m5outdir}/{benchmark}-{parsec_input}-{num_cpus}cpus"
                    args_list.append(deepcopy(args))
            else:
                args.outdir = f"{m5outdir}/{benchmark}-{args.parsec_input}-{num_cpus}cpus"
                args_list.append(deepcopy(args))

        else:
            args.outdir = f"{m5outdir}/{benchmark}-{num_cpus}cpus"
            args_list.append(deepcopy(args))

    return args_list
# configure_experiments() - end


def launch_experiments(args):
    """ Launch expeirments of all benchmarks for specified the schemes. """
    args_list = []

    if (args.launch_experiments == "violin_stage1"):
        if (1):
            profile_pc = ""
            violin_log_path = "./benchmarks/bin/gem-forge/omp_mv.s"
            with open(violin_log_path, "r") as logfile:
                found_pc = 0
                extract_done = 0
                for line in logfile:
                    if "__m512 valB = _mm512_load_ps(B + j);" in line:
                        found_pc += 1
                    elif ((found_pc == 1) & (extract_done == 0)):
                        line = line.split()
                        extract_done = 1
                        profile_pc = "0x" + line[0][:-1]
                        print(profile_pc)
                    else:
                        continue
            logfile.close()

            for runs in ["violin"]:
                args.routing = 1
                args.profile_llc = True
                args.profile_pc_low = profile_pc
                args.profile_pc_high = profile_pc
                args.profile_vaddr_low = "0"
                args.profile_vaddr_high = "0"
                args.l2_size = "256kB"
                args.llc_slice_size = "1MB"
                args.experiments_outdir = "AE-result/violin/link-128bits"
                args.link_width_bits = 128
                args.launch_experiments = runs
                args.num_cpus = 16

                if args.launch_experiments == "violin":
                    schemes = ["baseline"]

                args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"
                # Baseline
                if "baseline" in schemes:
                    args.coalescing = False #False
                    args.prepush = False
                    args.enable_multicast = False #False
                    args.prepush_filter = False
                    args.prepush_filter_nodrop = False
                    args.noc_coherence_constraint = "unordered"
                    baseline_list = configure_experiments(args, "baseline")

                if args.launch_experiments == "violin":
                    runs_per_scheme = len(baseline_list)

                    for i in range(runs_per_scheme):
                        args_list.append(baseline_list[i])

    if (args.launch_experiments == "violin_stage2"):
        if (1):
            profile_pc = ""
            violin_benchmark_path = "./benchmarks/bin/gem-forge/omp_mv.s"
            with open(violin_benchmark_path, "r") as logfile:
                found_pc = 0
                extract_done = 0
                for line in logfile:
                    if "__m512 valB = _mm512_load_ps(B + j);" in line:
                        found_pc += 1
                    elif ((found_pc == 1) & (extract_done == 0)):
                        line = line.split()
                        extract_done = 1
                        profile_pc = "0x" + line[0][:-1]
                        print(profile_pc)
                    else:
                        continue
            logfile.close()

            low_vaddr = ""
            high_vaddr = ""
            violin_log_path = "./m5out/AE-result/violin/link-128bits/baseline/mv-16cpus/sim.log"
            with open(violin_log_path, "r") as logfile:
                for line in logfile:
                    if "B vector base address:" in line:
                        line = line.split()
                        low_vaddr = line[4]
                        high_vaddr = line[8]
                        print(low_vaddr)
                        print(high_vaddr)
                        break
                    else:
                        continue
            logfile.close()

            for runs in ["violin"]:
                args.routing = 1
                args.profile_llc = True
                args.profile_pc_low = profile_pc
                args.profile_pc_high = profile_pc
                args.profile_vaddr_low = low_vaddr
                args.profile_vaddr_high = high_vaddr
                args.l2_size = "256kB"
                args.llc_slice_size = "1MB"
                args.experiments_outdir = "AE-result/violin/link-128bits"
                args.link_width_bits = 128
                args.launch_experiments = runs
                args.num_cpus = 16

                if args.launch_experiments == "violin":
                    schemes = ["baseline"]

                args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"
                # Baseline
                if "baseline" in schemes:
                    args.coalescing = False #False
                    args.prepush = False
                    args.enable_multicast = False #False
                    args.prepush_filter = False
                    args.prepush_filter_nodrop = False
                    args.noc_coherence_constraint = "unordered"
                    baseline_list = configure_experiments(args, "baseline")

                if args.launch_experiments == "violin":
                    runs_per_scheme = len(baseline_list)

                    for i in range(runs_per_scheme):
                        args_list.append(baseline_list[i])

    if (args.launch_experiments == "AE-all"):
        parsec_run_dir = f"{os.getcwd()}/benchmarks/runs/parsec"
        if not os.path.exists(parsec_run_dir):
            os.makedirs(parsec_run_dir)
            blackscholes_run_dir = f"{parsec_run_dir}/blackscholes"
            if not os.path.exists(blackscholes_run_dir):
                os.makedirs(blackscholes_run_dir)
            fluidanimate_run_dir = f"{parsec_run_dir}/fluidanimate"
            if not os.path.exists(fluidanimate_run_dir):
                os.makedirs(fluidanimate_run_dir)

        if (1):
            for runs in ["sensitivity"]:
                for sensitivity_type in ["TPC", "TimeWindow"]:
                    if sensitivity_type == "TPC":
                        for tpc in [4, 16, 64, 256, 512, 1024]:

                            args.allowed_window = 2000
                            args.feedback_threshold = tpc
                            args.l2_size = "256kB"
                            args.llc_slice_size = "1MB"
                            args.link_width_bits = 128
                            args.experiments_outdir = f"AE-result/256kB/link-128bits/TPC{tpc}-TimeWindow2000"
                            args.launch_experiments = runs
                            args.num_cpus = 16

                            if args.launch_experiments == "sensitivity":
                                schemes = ["prepush-multicast-feedback-restart-ratio"]

                            args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"
                            # Feedback
                            if "prepush-multicast-feedback-restart-ratio" in schemes:
                                args.coalescing = False
                                args.prepush = True
                                args.enable_multicast = True
                                args.prepush_filter = True
                                args.prepush_filter_nodrop = False
                                args.noc_coherence_constraint = "ordered-prepush-inv"
                                prepush_feedback_restart_ratio_list = configure_experiments(args, "prepush-multicast-feedback-restart-ratio")

                            if args.launch_experiments == "sensitivity":
                                runs_per_scheme = len(prepush_feedback_restart_ratio_list)

                                for i in range(runs_per_scheme):
                                    args_list.append(prepush_feedback_restart_ratio_list[i])
                    elif sensitivity_type == "TimeWindow":
                        for timewindow in [300, 400, 1000, 1500, 2500]:

                            args.allowed_window = timewindow
                            args.feedback_threshold = 16
                            args.l2_size = "256kB"
                            args.llc_slice_size = "1MB"
                            args.link_width_bits = 128
                            args.experiments_outdir = f"AE-result/256kB/link-128bits/TPC16-TimeWindow{timewindow}"
                            args.launch_experiments = runs
                            args.num_cpus = 16

                            if args.launch_experiments == "sensitivity":
                                schemes = ["prepush-multicast-feedback-restart-ratio"]

                            args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"
                            # Feedback
                            if "prepush-multicast-feedback-restart-ratio" in schemes:
                                args.coalescing = False
                                args.prepush = True
                                args.enable_multicast = True
                                args.prepush_filter = True
                                args.prepush_filter_nodrop = False
                                args.noc_coherence_constraint = "ordered-prepush-inv"
                                prepush_feedback_restart_ratio_list = configure_experiments(args, "prepush-multicast-feedback-restart-ratio")

                            if args.launch_experiments == "sensitivity":
                                runs_per_scheme = len(prepush_feedback_restart_ratio_list)

                                for i in range(runs_per_scheme):
                                    args_list.append(prepush_feedback_restart_ratio_list[i])

        if (1):
            for runs in ["cache-size"]:
                for cachesize in ["512kB", "1MB"]:

                    args.l2_size = cachesize
                    if (cachesize == "512kB"):
                        args.llc_slice_size = "1MB"
                        args.experiments_outdir = "AE-result/512kB/link-128bits"
                    elif (cachesize == "1MB"):
                        args.llc_slice_size = "2MB"
                        args.experiments_outdir = "AE-result/1024kB/link-128bits"
                    args.link_width_bits = 128
                    args.launch_experiments = runs
                    args.num_cpus = 16

                    if args.launch_experiments == "cache-size":
                        schemes = ["baseline", "bingo", "prepush-multicast-feedback-restart-ratio",
                                    "prepush-ack-multicast-feedback-restart-ratio"]

                    args.gem5 = f"{args.gem5_dir}/build/X86_MESI_Three_Level_PrepushAck_Bingo/gem5.opt"
                    # Bingo
                    if "bingo" in schemes:
                        args.coalescing = False #False
                        args.prepush = False #False
                        args.enable_multicast = False #False
                        args.prepush_filter = False #False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        bingo_list = configure_experiments(args, "bingo")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"
                    # Feedback
                    if "prepush-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        prepush_feedback_restart_ratio_list = configure_experiments(args, "prepush-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck_Feedback_Restart_Ratio/gem5.opt"
                    # Feedback
                    if "prepush-ack-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        prepush_ack_feedback_restart_ratio_list = configure_experiments(args, "prepush-ack-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"
                    # Baseline
                    if "baseline" in schemes:
                        args.coalescing = False #False
                        args.prepush = False
                        args.enable_multicast = False #False
                        args.prepush_filter = False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        baseline_list = configure_experiments(args, "baseline")

                    if args.launch_experiments == "cache-size":
                        runs_per_scheme = len(baseline_list)
                        assert len(bingo_list) == runs_per_scheme and \
                                len(prepush_feedback_restart_ratio_list) == runs_per_scheme and\
                                len(prepush_ack_feedback_restart_ratio_list) == runs_per_scheme

                        for i in range(runs_per_scheme):
                            args_list.append(baseline_list[i])
                            args_list.append(bingo_list[i])
                            args_list.append(prepush_feedback_restart_ratio_list[i])
                            args_list.append(prepush_ack_feedback_restart_ratio_list[i])

        if (1):
            for runs in ["link-study"]:
                for linkwidths in [64, 256, 512]:

                    args.l2_size = "256kB"
                    args.llc_slice_size = "1MB"
                    args.link_width_bits = linkwidths
                    args.experiments_outdir = f"AE-result/256kB/link-{args.link_width_bits}bits"
                    args.launch_experiments = runs
                    args.num_cpus = 16

                    if args.launch_experiments == "link-study":
                        schemes = ["baseline", "bingo", "prepush-multicast-feedback-restart-ratio",
                                "prepush-ack-multicast-feedback-restart-ratio"]

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"
                    # Feedback
                    if "prepush-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        prepush_feedback_restart_ratio_list = configure_experiments(args, "prepush-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck_Feedback_Restart_Ratio/gem5.opt"
                    # Feedback
                    if "prepush-ack-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        prepush_ack_feedback_restart_ratio_list = configure_experiments(args, "prepush-ack-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"
                    # Baseline
                    if "baseline" in schemes:
                        args.coalescing = False #False
                        args.prepush = False
                        args.enable_multicast = False #False
                        args.prepush_filter = False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        baseline_list = configure_experiments(args, "baseline")

                    args.gem5 = f"{args.gem5_dir}/build/X86_MESI_Three_Level_PrepushAck_Bingo/gem5.opt"

                    # Bingo
                    if "bingo" in schemes:
                        args.coalescing = False #False
                        args.prepush = False #False
                        args.enable_multicast = False #False
                        args.prepush_filter = False #False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        bingo_list = configure_experiments(args, "bingo")

                    if args.launch_experiments == "link-study":
                        runs_per_scheme = len(baseline_list)
                        assert len(prepush_feedback_restart_ratio_list) == runs_per_scheme and\
                                len(prepush_ack_feedback_restart_ratio_list) == runs_per_scheme and\
                                len(bingo_list) == runs_per_scheme

                        for i in range(runs_per_scheme):
                            args_list.append(baseline_list[i])
                            args_list.append(prepush_feedback_restart_ratio_list[i])
                            args_list.append(prepush_ack_feedback_restart_ratio_list[i])
                            args_list.append(bingo_list[i])

        if (1):
            for runs in ["all-speedup", "ablation"]:
                for loop_cpu in [16,64]:

                    args.l2_size = "256kB"
                    args.llc_slice_size = "1MB"
                    args.link_width_bits = 128
                    args.experiments_outdir = f"AE-result/256kB/link-128bits"
                    args.launch_experiments = runs
                    args.num_cpus = loop_cpu

                    if args.launch_experiments == "all-speedup":
                        schemes = ["baseline", "prepush-only","coalescing-multicast", "bingo",
                                    "prepush-multicast-feedback-restart-ratio",
                                    "prepush-ack-multicast-feedback-restart-ratio"]
                    elif args.launch_experiments == "ablation":
                        schemes = ["prepush-multicast", "prepush-multicast-filter"]

                    args.gem5 = f"{args.gem5_dir}/build/X86_MESI_Three_Level_PrepushAck_Bingo/gem5.opt"

                    # Bingo
                    if "bingo" in schemes:
                        args.coalescing = False #False
                        args.prepush = False #False
                        args.enable_multicast = False #False
                        args.prepush_filter = False #False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        args.scheme_name = f"bingo-{loop_cpu}cpus"
                        bingo_list = configure_experiments(args, "bingo")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"

                    # Feedback
                    if "prepush-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        args.scheme_name = f"prepush-multicast-feedback-restart-ratio-{loop_cpu}cpus"
                        prepush_feedback_restart_ratio_list = configure_experiments(args, "prepush-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck_Feedback_Restart_Ratio/gem5.opt"

                    # Feedback
                    if "prepush-ack-multicast-feedback-restart-ratio" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        args.scheme_name = f"prepush-ack-multicast-feedback-restart-ratio-{loop_cpu}cpus"
                        prepush_ack_feedback_restart_ratio_list = configure_experiments(args, "prepush-ack-multicast-feedback-restart-ratio")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"

                    # Baseline
                    if "baseline" in schemes:
                        args.coalescing = False #False
                        args.prepush = False
                        args.enable_multicast = False #False
                        args.prepush_filter = False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        args.scheme_name = f"baseline-{loop_cpu}cpus"
                        baseline_list = configure_experiments(args, "baseline")

                    # Coalescing-Multicast
                    if "coalescing-multicast" in schemes:
                        args.coalescing = True
                        args.enable_multicast = True
                        args.prepush = False
                        args.prepush_filter = False
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "unordered"
                        args.scheme_name = f"coalescing-multicast-{loop_cpu}cpus"
                        coalescing_multicast_list = \
                                configure_experiments(args, "coalescing-multicast")

                    args.gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush/gem5.opt"

                    if "prepush-only" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = False
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = True
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        args.scheme_name = f"prepush-only-{loop_cpu}cpus"
                        prepush_only_list = configure_experiments(args, "prepush-only")

                    if "prepush-multicast" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True
                        args.prepush_filter = True
                        args.prepush_filter_nodrop = True
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        args.scheme_name = f"prepush-multicast-{loop_cpu}cpus"
                        prepush_multicast_list = configure_experiments(args, "prepush-multicast")

                    # Prepush-Multicast-Filtering with ordered Prepush-Inv NoC
                    if "prepush-multicast-filter" in schemes:
                        args.coalescing = False
                        args.prepush = True
                        args.enable_multicast = True #True
                        args.prepush_filter = True #True
                        args.prepush_filter_nodrop = False
                        args.noc_coherence_constraint = "ordered-prepush-inv"
                        args.scheme_name = f"prepush-multicast-filter-{loop_cpu}cpus"
                        prepush_multicast_filter_list = \
                                configure_experiments(args, "prepush-multicast-filter")

                    if args.launch_experiments == "all-speedup":
                        runs_per_scheme = len(baseline_list)
                        assert len(prepush_only_list) == runs_per_scheme and \
                                len(coalescing_multicast_list) == runs_per_scheme and \
                                len(bingo_list) == runs_per_scheme and \
                                len(prepush_feedback_restart_ratio_list) == runs_per_scheme and\
                                len(prepush_ack_feedback_restart_ratio_list) == runs_per_scheme

                        for i in range(runs_per_scheme):
                            args_list.append(baseline_list[i])
                            args_list.append(prepush_only_list[i])
                            args_list.append(coalescing_multicast_list[i])
                            args_list.append(bingo_list[i])
                            args_list.append(prepush_feedback_restart_ratio_list[i])
                            args_list.append(prepush_ack_feedback_restart_ratio_list[i])

                    elif args.launch_experiments == "ablation":
                        runs_per_scheme = len(prepush_multicast_list)
                        assert len(prepush_multicast_filter_list) == runs_per_scheme

                        for i in range(runs_per_scheme):
                            args_list.append(prepush_multicast_list[i])
                            args_list.append(prepush_multicast_filter_list[i])

    if args.sweep_thread_pool_size is None:
        args.sweep_thread_pool_size = mp.cpu_count() * 4 // 5

    if len(args_list) < args.sweep_thread_pool_size:
        args.sweep_thread_pool_size = len(args_list)

    print(len(args_list))
    filename = f"./arglist.txt"
    with open(filename, 'w') as f:
        for i in range(len(args_list)):
            f.write(json.dumps(vars(args_list[i])) + "\n")
    f.close()

    pool = mp.Pool(args.sweep_thread_pool_size)
    pool.map(run_gem5_instance, args_list)
    pool.close()
    pool.join()

    print("Launched all simulation jobs!")


def main():

    parser = argparse.ArgumentParser(
            description="A script for launching gem5 simulation")

    parser.add_argument("--gem5", type=str,
                        default="./build/X86_MESI_Two_Level/gem5.opt",
                        help="gem5 binary: path for gem5.opt or gem5.debug "
                             "[Default: ./build/X86_MESI_Two_Level/gem5.opt]")
    parser.add_argument("--gem5-dir", type=str,
                        default="./gem5",
                        help="gem5 source directory [Default: ./gem5]")
    parser.add_argument("--outdir", default="m5out", type=str,
                        help="Set the output directory [Default: m5out]")
    parser.add_argument("--no-listener", default=False, action="store_true",
                        help="Disable listener mode [Default: False]")
    parser.add_argument("--cpu-type", type=str, default="TunedCPU",
                        choices=["VerbatimCPU", "TunedCPU",
                                 "UnconstrainedCPU"],
                        help="CPU models, VerbatimCPU is the Skylake "
                             "documented micro-architecture, TunedCPU is the "
                             "tuned configuration to match the performance "
                             "with real Skylake hardware, UnconstrainedCPU "
                             "is configured with maximum pipeline widths and "
                             "minimum delays, [Default: TunedCPU].")
    parser.add_argument("--num-cpus", default=16, type=int,
                        help="Number of CPUs [Default: 16]")
    parser.add_argument("--l2-size", default="256kB", type=str,
                        help="Set the private L2 cache size [Default: 256kB]")
    parser.add_argument("--llc-slice-size", default="1MB", type=str,
                        help="Set the LLC slice cache size [Default: 1MB]")
    parser.add_argument("--ruby-clock", default="2GHz", type=str,
                        help="Clock frequency for ruby cache hierarchy, "
                             "[Default: 2GHz]")
    parser.add_argument("--scheme-name", default=None, type=str,
                        help="Set a scheme name to generate output files with"
                             " the scheme name for parsec benchmark options "
                             "[Default: None]")
    parser.add_argument("--benchmark", default="cachebw", type=str,
                        choices=["cachebw", "readbw_multilevel", "mlp",
                                 "conv3dnofoowarm", "conv3dfoowarm", "mv",
                                 "backprop", "bfs", "btree", "cfd", "hotspot",
                                 "hotspot3D", "kmeans", "lud", "nn", "nw",
                                 "particlefilter", "pathfinder", "srad",
                                 "blackscholes", "bodytrack", "canneal",
                                 "dedup", "facesim", "ferret", "fluidanimate",
                                 "freqmine", "raytrace", "streamcluster",
                                 "swaptions", "vips", "x264","covertchannel"],
                        help="Benchmark to run [Default: cachebw]")
    parser.add_argument("--benchmarks", type=str, nargs='*',
                        default=["cachebw", "readbw_multilevel", "mlp", "conv3dfoowarm", "mv",
                                 "backprop", "bfs", "lud", "particlefilter", "pathfinder", "blackscholes",
                                 "bodytrack", "fluidanimate", "freqmine", "swaptions"],
                        help="benchmark list [Default: ['cachebw', 'readbw_multilevel', 'mlp', "
                             "'conv3dnofoowarm', 'conv3dfoowarm', 'mv', 'backprop', 'bfs', 'btree', "
                             "'cfd', 'hotspot', 'hotspot3D', 'lud',"
                             " 'nn', 'nw', " "'particlefilter', 'pathfinder',"
                             " 'srad', 'blackscholes']]")
    parser.add_argument("--parsec-input", default="test", type=str,
                        help="PARSEC input: test, small, medium, large "
                             "[Default: test]")
    parser.add_argument("--parsec-input-list", default=None, type=str,
                        nargs="*",
                        help="parsec input list: [test, small, medium], "
                             "[Default: None]")
    parser.add_argument("--cfd-iterations", default=0, type=int,
                        help="Number of iterations as input to cfd")
    parser.add_argument("--particlefilter-frames", default=0, type=int,
                        help="Number of frames as input to particlefilter")
    parser.add_argument("--lud-size", default=1024, type=int, #1024
                        help="Matrix size for LU Decomposition")
    parser.add_argument("--test-input", default=False, action="store_true",
                        help="Use test input for benchmark testing")
    parser.add_argument("--window-cycles", default=None, type=int,
                        help="sharer characterizatioon time window "
                             "[Default: None]")
    parser.add_argument("--log", default=False, action="store_true",
                        help="Redirect stdout and stderr to a sim.log in "
                             "outdir [Default: False]")
    parser.add_argument("--dry-run", default=False, action="store_true",
                        help="Print command ony [Default: False]")
    parser.add_argument("--sweep", default=False, action="store_true",
                        help="Sweep configuration parameters")
    parser.add_argument("--sweep-num-cpus", type=int, nargs="*", default=[16],
                        help="Sweep number of cpus [Default: [16]]")
    parser.add_argument("--sweep-window-cycles",
                        type=int, nargs="*", default=[200],
                        help="Sweep window cycles [Default: [200]]")
    parser.add_argument("--sweep-thread-pool-size", type=int, default=None,
                        help="Number of threads for sweep simulations "
                             "[Default: half of the cpus in the system]")
    parser.add_argument("--debug-flags", metavar="FLAG[,FLAG]",
                        type=str, default="PseudoInst",
                        help="Sets the flags for debug output (-FLAG desables"
                             " a flag) [Default: PseudoInst]")
    parser.add_argument("--debug-start", metavar="TICK",
                        type=int, default=None,
                        help="Start debug output at TICK")
    parser.add_argument("--debug-end", metavar="TICK", type=int, default=None,
                        help="End debug output at TICK")
    parser.add_argument("--debug-file", metavar="FILE", default=None,
                        help="Sets the output file for debug [Default: None]")
    parser.add_argument("--profile-pc-low", default=None, type=str,
                        help="The low PC boundry for profile PC range, can "
                             "be either decimal or hex form [Default: None]")
    parser.add_argument("--profile-pc-high", default=None, type=str,
                        help="The high PC boundry for profile PC range, can "
                             "be either decimal or hex form [Default: None]")
    parser.add_argument("--profile-vaddr-low", default=None, type=str,
                        help="The low addr boundry for profile data range,"
                             "can be either decimal or hex [Default: None]")
    parser.add_argument("--profile-vaddr-high", default=None, type=str,
                        help="The high addr boundry for profile data range, "
                             "can be either decimal or hex [Default: None]")
    parser.add_argument("--profile-llc", default=False, action="store_true",
                        help="Enable LLC sharers profiling")
    parser.add_argument("--message-buffer-size", default=0, type=int,
                        help="Message buffer size, default is 0 for infinite")
    parser.add_argument("--enable-prefetch", default=False,
                        action="store_true",
                        help="Enable ruby prefetcher [Default: False]")
    parser.add_argument("--prepush", default=False, action="store_true",
                        help="Enable prepush.")
    parser.add_argument("--always-prepush", default=False, action="store_true",
                        help="Prepush upon a shared data request if prepush "
                             "is enabled, o.w., only prepush once.")
    parser.add_argument("--enable-multicast", default=False,
                        action="store_true",
                        help="enable multicast hardware support.")
    parser.add_argument("--prepush-filter", default=False,
                        action="store_true",
                        help="enable prepush in-network filteriing.")
    parser.add_argument("--profile-prepush", default=False,
                        action="store_true",
                        help="Profile the prepush initiated PCs for their "
                             "counts and the total number of sharers")
    parser.add_argument("--coalescing", default=False, action="store_true",
                        help="enable coalescing to multicast to all the "
                             "outstanding GetS requestors")
    parser.add_argument("--link-width-bits", type=int, default=128,
                        help="Network link width in bits")
    parser.add_argument("--routing", default=4, type=int,
                        help="""routing algorithm in network.
                            0: weight-based table,
                            1: XY (for Mesh),
                            2: Custom,
                            3: YX (for Mesh),
                            4: XY-YX (for Mesh)
                            5: Adaptive (for Mesh)""")
    parser.add_argument("--hold-switch-for-multicast-only",
                        action="store_true", default=False,
                        help="Hold switch for multicast packets only but not "
                             "other data packets")
    parser.add_argument("--noc-coherence-constraint", type=str,
                        default="unordered",
                        choices=["unordered", "ordered-vnet",
                                 "ordered-prepush-inv"],
                        help="NoC coherence traffic ordering constraint (for "
                             "prepush): 'unordered' for unordered, "
                             "'ordered-vnet' for ordered response and "
                             "unblock-forward virtual networks, "
                             "'ordered-prepush-inv' for only ordered prepush "
                             "and invalidation messages")
    parser.add_argument("--launch-experiments", default=None, type=str,
                        choices = ["all", "all-speedup", "ablation", "link-study", "cache-size", "sensitivity", "violin_stage1",
                                   "baseline", "prepush-only", "prepush-multicast", "prepush-multicast-filter", "violin_stage2",
                                   "AE-all", "coalescing-multicast", "bingo", "prepush-multicast-feedback-restart-ratio",
                                   "prepush-ack-multicast-feedback-restart-ratio"],
                        help="launch batches of experiments, choose one from "
                             "the choices, default: None")
    parser.add_argument("--experiments-outdir", default="experiments",
                        type=str,
                        help="Set the output directory [Default: "
                             "experiments (m5out/experiments)]")
    # TODO: add prepush option and decouple it from debug-start and debug-end

    args = parser.parse_args()

    if args.launch_experiments is None:
        if not os.path.exists(args.gem5):
            print(f"Error: {args.gem5} not exists!")
            print("Please specify a valid gem5 binary")
            return
        else:
            args.gem5 = f"{os.getcwd()}/{args.gem5}"
            assert args.gem5_dir in args.gem5
            assert os.path.exists(args.gem5)

    if args.sweep:
        sweep(args)
    elif args.launch_experiments is not None:
        temp = args.launch_experiments
        gem5 = f"{args.gem5_dir}/build/X86_MESI_Three_Level_PrepushAck_Bingo/gem5.opt"
        if not os.path.exists(gem5) and \
                temp in ["all", "all-speedup", "cache-size", "bingo"]:
            print(f"Error: {gem5} not exists!")
            return
        gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush/gem5.opt"
        if not os.path.exists(gem5) and \
                temp in ["all", "all-speedup", "ablation", "prepush-only", "prepush-multicast", "prepush-multicast-filter"]:
            print(f"Error: {gem5} not exists!")
            return
        gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck/gem5.opt"
        if not os.path.exists(gem5) and \
                temp in ["all", "all-speedup", "link-study", "violin", "cache-size", "baseline", "coalescing-multicast"]:
            print(f"Error: {gem5} not exists!")
            return
        gem5 = f"./gem5/build/X86_MESI_Three_Level_Prepush_Feedback_Restart_Ratio/gem5.opt"
        if not os.path.exists(gem5) and \
                temp in ["all", "all-speedup", "link-study", "cache-size", "sensitivity", "prepush-multicast-feedback-restart-ratio"]:
            print(f"Error: {gem5} not exists!")
            return
        gem5 = f"./gem5/build/X86_MESI_Three_Level_PrepushAck_Feedback_Restart_Ratio/gem5.opt"
        if not os.path.exists(gem5) and \
                temp in ["all", "all-speedup", "link-study", "cache-size", "prepush-ack-multicast-feedback-restart-ratio"]:
            print(f"Error: {gem5} not exists!")
            return
        launch_experiments(args)
    else:
        run_gem5_instance(args)


if __name__ == "__main__":
    main()
