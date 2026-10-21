CC      ?= gcc
AR      ?= ar
STD     := -std=c11
WARN    := -Wall -Wextra -Werror
OPT     ?= -O2

ifeq ($(NATIVE),1)
OPT += -march=native
endif

OPENMP  :=

FPFLAGS := -ffp-contract=off -DSAG_FP_CONTRACT_OFF=1

SRC_DIR   := src
INC_DIR   := include
TEST_DIR  := tests
BUILD_DIR := build

BIN      := sagsep
TEST_BIN := sagsep_tests
LIB      := libsagsep.a

HAVE_TESTS := $(if $(wildcard $(TEST_DIR)),1,)

INCS := -I$(INC_DIR) -iquote $(SRC_DIR)
DEFS :=
LIBS := -lm

# ---------------------------------------------------------------------------
# Gurobi (required; header and library paths are derived from GUROBI_HOME)
# ---------------------------------------------------------------------------
GUROBI_HOME    ?=
GUROBI_INCLUDE ?= $(GUROBI_HOME)/include
GUROBI_LIBDIR  ?= $(GUROBI_HOME)/lib
GUROBI_SO      := $(filter-out %_light.so,$(wildcard $(GUROBI_LIBDIR)/libgurobi*.so))
GUROBI_LIB     ?= $(patsubst lib%,%,$(basename $(notdir $(lastword $(sort $(GUROBI_SO))))))

ifneq ($(strip $(GUROBI_HOME))$(strip $(GUROBI_INCLUDE)),)
ifneq ($(wildcard $(GUROBI_INCLUDE)/gurobi_c.h),)
ifneq ($(strip $(GUROBI_LIB)),)
HAVE_GUROBI := 1
endif
endif
endif

ifdef HAVE_GUROBI
DEFS += -DHAVE_GUROBI=1
INCS += -I$(GUROBI_INCLUDE)
LIBS += -L$(GUROBI_LIBDIR) -l$(GUROBI_LIB) -Wl,-rpath,$(GUROBI_LIBDIR)
endif

BUILD_GOALS := all $(BIN) $(TEST_BIN) $(LIB) sagsep sagsep_tests libsagsep.a \
               tests tests-strict lib
ifneq ($(filter $(BUILD_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)),)
ifneq ($(wildcard $(SRC_DIR)/lp.c),)
ifndef HAVE_GUROBI
$(error Gurobi not found. Set GUROBI_HOME to a Gurobi installation with \
include/gurobi_c.h and lib/libgurobi*.so and a valid license, for example \
`make GUROBI_HOME=/path/to/gurobi`. GUROBI_INCLUDE, GUROBI_LIBDIR and \
GUROBI_LIB can be overridden individually if the installation is laid out \
differently. Looked for a header at "$(GUROBI_INCLUDE)/gurobi_c.h" and for \
libgurobi*.so in "$(GUROBI_LIBDIR)")
endif
endif
endif

# ---------------------------------------------------------------------------
# onnxruntime
# ---------------------------------------------------------------------------
ONNXRUNTIME_HOME ?=
ifneq ($(strip $(ONNXRUNTIME_HOME)),)
ifneq ($(wildcard $(ONNXRUNTIME_HOME)/include/onnxruntime_c_api.h),)
HAVE_ONNXRUNTIME := 1
endif
endif

ifdef HAVE_ONNXRUNTIME
DEFS += -DHAVE_ONNXRUNTIME=1
INCS += -I$(ONNXRUNTIME_HOME)/include
LIBS += -L$(ONNXRUNTIME_HOME)/lib -lonnxruntime -Wl,-rpath,$(ONNXRUNTIME_HOME)/lib
endif

# ---------------------------------------------------------------------------
# Source files (a new module joins the build as soon as its file lands)
# ---------------------------------------------------------------------------
CORE_SRC := $(sort $(filter-out $(SRC_DIR)/main.c,\
                     $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/families/*.c) \
                     $(wildcard $(SRC_DIR)/families/tts/*.c)))
ifndef HAVE_ONNXRUNTIME
CORE_SRC := $(filter-out $(SRC_DIR)/forward_onnx.c,$(CORE_SRC))
endif

CORE_OBJ := $(CORE_SRC:%.c=$(BUILD_DIR)/%.o)
MAIN_OBJ := $(BUILD_DIR)/$(SRC_DIR)/main.o

ifdef HAVE_TESTS
TEST_SRC := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJ := $(TEST_SRC:%.c=$(BUILD_DIR)/%.o)
else
TEST_SRC :=
TEST_OBJ :=
endif

EXTRA_CFLAGS ?=

CFLAGS  := $(STD) $(OPT) $(FPFLAGS) $(OPENMP) $(WARN) $(INCS) $(DEFS) $(EXTRA_CFLAGS) -MMD -MP
LDFLAGS := $(OPENMP)

TEST_CFLAGS :=

.PHONY: all lib tests tests-strict clean build-info help
.SUFFIXES:

ifdef HAVE_TESTS
all: $(BIN) $(TEST_BIN)
else
all: $(BIN)
endif

$(BIN): $(CORE_OBJ) $(MAIN_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

lib: $(LIB)

$(LIB): $(CORE_OBJ)
	rm -f $@
	$(AR) rcs $@ $^

ifdef HAVE_TESTS
$(TEST_BIN): $(CORE_OBJ) $(TEST_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TEST_CFLAGS) -c -o $@ $<

tests: $(TEST_BIN)
	./$(TEST_BIN)

tests-strict: $(TEST_BIN)
	./$(TEST_BIN) --strict
else
tests tests-strict:
	@echo "no $(TEST_DIR)/ directory in this tree -- nothing to test"
endif

build-info:
	@echo "CC              = $(CC)"
	@echo "CFLAGS          = $(CFLAGS)"
	@echo "LIBS            = $(LIBS)"
	@echo "GUROBI_HOME     = $(GUROBI_HOME)"
	@echo "GUROBI_INCLUDE  = $(GUROBI_INCLUDE)"
	@echo "GUROBI_LIBDIR   = $(GUROBI_LIBDIR)"
	@echo "GUROBI_LIB      = $(GUROBI_LIB)"
	@echo "HAVE_GUROBI     = $(if $(HAVE_GUROBI),yes,no)"
	@echo "ONNXRUNTIME_HOME= $(ONNXRUNTIME_HOME)"
	@echo "HAVE_ONNXRUNTIME= $(if $(HAVE_ONNXRUNTIME),yes,no)"
	@echo "HAVE_TESTS      = $(if $(HAVE_TESTS),yes,no)"
	@echo "NATIVE          = $(if $(filter 1,$(NATIVE)),yes,no)"
	@echo "CORE_SRC        = $(CORE_SRC)"
	@echo "TEST_SRC        = $(TEST_SRC)"

help:
	@echo "targets: all | lib | tests | tests-strict | build-info | clean"
	@echo "options: NATIVE=1 GUROBI_HOME=... ONNXRUNTIME_HOME=... OPT=... EXTRA_CFLAGS=..."

clean:
	rm -rf $(BUILD_DIR) $(BIN) $(TEST_BIN) $(LIB)

-include $(CORE_OBJ:.o=.d) $(MAIN_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
