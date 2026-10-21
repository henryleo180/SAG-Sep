CXX      ?= g++
CXXFLAGS ?= -O3 -fPIC -w
AR       ?= ar

SRCS := $(sort $(wildcard *.cpp))
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

SHARED := libcvrpsep.so
STATIC := libcvrpsep.a

.PHONY: all shared static clean
all: shared static

shared: $(SHARED)
static: $(STATIC)

$(SHARED): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS)

$(STATIC): $(OBJS)
	@rm -f $@
	$(AR) rcs $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I. -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(SHARED) $(STATIC)

-include $(DEPS)
