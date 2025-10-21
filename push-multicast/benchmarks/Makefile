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
