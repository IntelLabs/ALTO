CXX       = g++
LINKER   = $(CC)

OPENMP   = -fopenmp
ifeq ($(BLAS_LIBRARY),MKL)
BLASCFLAGS = -DMKL_ILP64
BLASINC  = -I$(MKLROOT)/include
BLASLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5
else
BLASCFLAGS =
BLASINC  =
BLASLIBS = -lopenblas -lgfortran
endif

VERSION  = --version
ifeq ($(DEBUG), true)
CFLAGS   = -O0 -static-libasan -O -g -fsanitize=address -fno-omit-frame-pointer -march=native -static -Wall -g -std=c++17 -D _GLIBCXX_PARALLEL 
CXXFLAGS += $(OPENMP) -DMKL_ILP64 -O0 -static-libasan -O -g -fsanitize=address -fno-omit-frame-pointer -march=native -static -Wall -g -std=c++17 -D _GLIBCXX_PARALLEL 
LIBS	 = -static-libasan -O -g -fsanitize=address -fno-omit-frame-pointer -lpthread -lm -ldl $(BLASLIBS)
else
CFLAGS   = -O3 -march=native -static -Wall -g -std=c++17 -D _GLIBCXX_PARALLEL 
#CFLAGS  += -T/opt/FJSVxos/mmm/util/bss-2mb.lds -L/opt/FJSVxos/mmm/lib64 -lmpg
CXXFLAGS += $(OPENMP) -DMKL_ILP64
LIBS	 = -Wl,--no-as-needed -lpthread -lm -ldl $(BLASLIBS)
endif

LFLAGS   = $(OPENMP)
DEFINES  = -DALTO_MASK_LENGTH=$(ALTO_MASK_LENGTH) -DMAX_NUM_MODES=$(MAX_NUM_MODES)
INCLUDES = $(BLASINC)
