CXX      = icpc
LINKER   = $(CC)

OPENMP   = -qopenmp
ifeq ($(BLAS_LIBRARY),MKL)
BLASLFLAGS 	= -mkl=parallel
INCLUDFLAGS	= 
BLASLIBS 	= -liomp5 -lpthread -lm -ldl #-lmkl_intel_lp64 #-lmkl_intel_thread
else
BLASLFLAGS 	= 
INCLUDFLAGS	= 
BLASLIBS 	= -lopenblas
endif

VERSION  = --version
CXXFLAGS = -O3 -xHost -static -Wall -g -std=c++17 -D _GLIBCXX_PARALLEL $(OPENMP) -qopt-zmm-usage=high -funroll-loops -fstrict-aliasing #-ipo
LFLAGS   = $(OPENMP) $(BLASLFLAGS)
DEFINES  = -DALTO_MASK_LENGTH=$(ALTO_MASK_LENGTH) -DMAX_NUM_MODES=$(MAX_NUM_MODES)
INCLUDES = $(INCLUDEFLAGS)
LIBS     = $(BLASLIBS)
