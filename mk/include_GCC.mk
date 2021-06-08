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

ifneq ($(ALTERNATIVE_PEXT),true)
CXXFLAGS += -mbmi2
endif

VERSION  = --version
ifeq ($(DEBUG),true)
CXXFLAGS += $(OPENMP) $(BLASCFLAGS) -O0 -static-libasan -O -g -fsanitize=address -fno-omit-frame-pointer -march=native -static -Wall -g -std=c++17 -D _GLIBCXX_PARALLEL
LIBS	 = -static-libasan -O -g -fsanitize=address -fno-omit-frame-pointer -lpthread -lm -ldl $(BLASLIBS)
else
CXXFLAGS += $(OPENMP) $(BLASCFLAGS) -O3 -march=native -static -g -std=c++17 -D_GLIBCXX_PARALLEL
LIBS	 = -Wl,--no-as-needed -lpthread -lm -ldl $(BLASLIBS)
endif

LFLAGS   = $(OPENMP)
DEFINES  = -DALTO_MASK_LENGTH=$(ALTO_MASK_LENGTH) -DMAX_NUM_MODES=$(MAX_NUM_MODES)
INCLUDES = $(BLASINC)
