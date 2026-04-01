# NexaLang Runtime — Build the C extensions
#
# Usage:
#   make              — build libturboquant.so (Linux) or libturboquant.dylib (macOS)
#   make clean        — remove build artifacts
#   make test         — compile and run a quick C-level smoke test

CC       ?= cc
CFLAGS   := -O2 -fPIC -std=c11 -Wall -Wextra
UNAME_S  := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    SHARED_FLAGS := -dynamiclib
else
    SHARED_EXT := so
    SHARED_FLAGS := -shared
endif

LIB := libturboquant.$(SHARED_EXT)

all: $(LIB)

$(LIB): turboquant.c turboquant.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ turboquant.c -lm

clean:
	rm -f $(LIB) test_tq

test: $(LIB)
	@echo "=== TurboQuant C smoke test ==="
	$(CC) $(CFLAGS) -o test_tq -DTURBOQUANT_TEST turboquant_test.c -L. -lturboquant -lm -Wl,-rpath,.
	./test_tq

.PHONY: all clean test
