# CONFIGURE BUILD SYSTEM
TARGET     = cpd$(ALTO_MASK_LENGTH)
BUILD_DIR  = ./build-$(ALTO_MASK_LENGTH)
INC_DIR    = ./include
SRC_DIR    = ./src
MAKE_DIR   = ./mk
Q         ?= @


##########################################
# DO NOT EDIT BELOW
include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk

space := $(eval) $(eval)
ifneq ($(strip $(MODES_SPECIALIZED)),0)
  MODES_SPECIALIZED := 0,$(MODES_SPECIALIZED)
endif
ifneq ($(strip $(RANKS_SPECIALIZED)),0)
  RANKS_SPECIALIZED := 0,$(RANKS_SPECIALIZED)
endif
DEFINES += -DALTO_MODES_SPECIALIZED=$(subst $(space),,$(MODES_SPECIALIZED))
DEFINES += -DALTO_RANKS_SPECIALIZED=$(subst $(space),,$(RANKS_SPECIALIZED))
ifeq ($(THP_PRE_ALLOCATION),true)
DEFINES += -DALTO_PRE_ALLOC
endif
ifeq ($(strip $(ALTERNATIVE_PEXT)),true)
DEFINES += -DALT_PEXT
endif
ifeq ($(strip $(MEMTRACE)),true)
DEFINES += -DALTO_MEM_TRACE
endif
ifeq ($(strip $(DEBUG)),true)
DEFINES += -DALTO_DEBUG
endif
ifeq ($(strip $(BLAS_LIBRARY)),MKL)
DEFINES += -DMKL
endif
ifeq ($(strip $(ALTO_IDX_TYPE)),INT32)
DEFINES += -DIDX_TYPE32
endif
ifeq ($(strip $(ALTO_FP_TYPE)),FP32)
DEFINES += -DFP_TYPE32
endif
ifeq ($(strip $(ALTO_VAL_TYPE)),INT32)
DEFINES += -DVAL_TYPE_INT32
else ifeq ($(strip $(ALTO_VAL_TYPE)),INT64)
DEFINES += -DVAL_TYPE_INT64
endif

INCLUDES += -I$(INC_DIR)

SRC        = $(wildcard $(SRC_DIR)/*.cpp)
ASM        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.s,$(SRC))
OBJ        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o,$(SRC))
CPPFLAGS  := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)


$(TARGET): $(BUILD_DIR) $(OBJ)
	@echo "===>  LINKING  $(TARGET)"
	$(Q)$(CXX) $(LFLAGS) -o $(TARGET) $(OBJ) $(LIBS)

asm: $(BUILD_DIR) $(ASM)

info:
	@echo $(CXXFLAGS)
	$(Q)$(CXX) $(VERSION)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | build_dir
	$(Q)$(CXX) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $@

$(BUILD_DIR)/%.o:  $(SRC_DIR)/%.cpp
	@echo "===>  COMPILE  $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/%.s:  $(SRC_DIR)/%.cpp
	@echo "===>  GENERATE ASM  $@"
	$(CXX) -S $(CPPFLAGS) $(CXXFLAGS) $< -o $@

.PHONY: build_dir
build_dir: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f $(TARGET)
