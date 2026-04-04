# NexaLang Runtime — Build the C extensions
#
# Usage:
#   make              — build all runtime libraries
#   make clean        — remove build artifacts
#   make test         — compile and run C-level smoke tests

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

LIB_TQ    := libturboquant.$(SHARED_EXT)
LIB_ASYNC := libnexa_async.$(SHARED_EXT)

all: $(LIB_TQ) $(LIB_ASYNC)

$(LIB_TQ): turboquant.c turboquant.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ turboquant.c -lm

$(LIB_ASYNC): nexa_async.c
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ nexa_async.c

clean:
	rm -f $(LIB_TQ) $(LIB_ASYNC) test_tq

test: $(LIB_TQ)
	@echo "=== TurboQuant C smoke test ==="
	$(CC) $(CFLAGS) -o test_tq -DTURBOQUANT_TEST turboquant_test.c -L. -lturboquant -lm -Wl,-rpath,.
	./test_tq

.PHONY: all clean test
