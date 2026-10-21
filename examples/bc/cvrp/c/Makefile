SAGSEP_C ?= ../../../../package/c
COMMON   ?= ../../common/c
CVRPSEP  ?= ../../../third_party/cvrpsep

CC   ?= gcc
STD  := -std=c11
WARN := -Wall -Wextra -Werror
OPT  ?= -O2

FPFLAGS := -ffp-contract=off

BIN        := bc_cvrp
BUILD_DIR  := build
SRC        := main.c bc.c formulation.c vrp_io.c
COMMON_SRC := instance.c separator_cvrpsep.c separator_sagsep.c util.c
OBJ        := $(SRC:%.c=$(BUILD_DIR)/%.o) $(COMMON_SRC:%.c=$(BUILD_DIR)/%.o)

SAGSEP_LIB  := $(SAGSEP_C)/libsagsep.a
CVRPSEP_LIB := $(CVRPSEP)/libcvrpsep.a

# ---------------------------------------------------------------------------
# Gurobi (required)
# ---------------------------------------------------------------------------
GUROBI_HOME    ?=
GUROBI_INCLUDE ?= $(GUROBI_HOME)/include
GUROBI_LIBDIR  ?= $(GUROBI_HOME)/lib
GUROBI_SO      := $(filter-out %_light.so,$(wildcard $(GUROBI_LIBDIR)/libgurobi*.so))
GUROBI_LIB     ?= $(patsubst lib%,%,$(basename $(notdir $(lastword $(sort $(GUROBI_SO))))))

BUILD_GOALS := all libs $(BIN)
ifneq ($(filter $(BUILD_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)),)
ifeq ($(wildcard $(SAGSEP_C)/include/sep_api.h),)
$(error SAG-Sep engine not found at "$(SAGSEP_C)". Set SAGSEP_C to the engine \
tree, which must contain include/sep_api.h and a Makefile with a `lib` target, \
for example `make SAGSEP_C=/path/to/SAG-Sep/package/c`)
endif
ifeq ($(wildcard $(GUROBI_INCLUDE)/gurobi_c.h),)
$(error Gurobi not found. Set GUROBI_HOME to an installation with \
include/gurobi_c.h and lib/libgurobi*.so, for example \
`make GUROBI_HOME=/path/to/gurobi`. GUROBI_INCLUDE, GUROBI_LIBDIR and \
GUROBI_LIB can be overridden individually. Looked for a header at \
"$(GUROBI_INCLUDE)/gurobi_c.h")
endif
endif

# ---------------------------------------------------------------------------
# onnxruntime (only the engine archive references it)
# ---------------------------------------------------------------------------
ONNXRUNTIME_HOME ?=
ONNX_LIBS :=
ifneq ($(strip $(ONNXRUNTIME_HOME)),)
ONNX_LIBS := -L$(ONNXRUNTIME_HOME)/lib -lonnxruntime -Wl,-rpath,$(ONNXRUNTIME_HOME)/lib
endif

INCS := -I$(SAGSEP_C)/include -I$(COMMON) -isystem $(CVRPSEP) -isystem $(GUROBI_INCLUDE)

CFLAGS  := $(STD) $(OPT) $(FPFLAGS) $(WARN) $(INCS) -MMD -MP
LDLIBS  := -L$(GUROBI_LIBDIR) -l$(GUROBI_LIB) -Wl,-rpath,$(GUROBI_LIBDIR) \
           $(ONNX_LIBS) -lstdc++ -lm

.PHONY: all libs clean build-info
.SUFFIXES:

all: $(BIN)

libs:
	$(MAKE) -C $(SAGSEP_C) lib GUROBI_HOME=$(GUROBI_HOME) ONNXRUNTIME_HOME=$(ONNXRUNTIME_HOME)
	$(MAKE) -C $(CVRPSEP) static

$(BIN): $(OBJ) | libs
	$(CC) -o $@ $(OBJ) $(SAGSEP_LIB) $(CVRPSEP_LIB) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(COMMON)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

build-info:
	@echo "CC               = $(CC)"
	@echo "CFLAGS           = $(CFLAGS)"
	@echo "LDLIBS           = $(LDLIBS)"
	@echo "SAGSEP_C         = $(SAGSEP_C)"
	@echo "COMMON           = $(COMMON)"
	@echo "CVRPSEP          = $(CVRPSEP)"
	@echo "GUROBI_HOME      = $(GUROBI_HOME)"
	@echo "GUROBI_LIB       = $(GUROBI_LIB)"
	@echo "ONNXRUNTIME_HOME = $(ONNXRUNTIME_HOME)"

clean:
	rm -rf $(BUILD_DIR) $(BIN)

-include $(OBJ:.o=.d)
