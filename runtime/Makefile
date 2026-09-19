# Native runtime build outputs stay outside the source directory.
CC ?= cc
CFLAGS ?= -O2 -fPIC -std=c11 -Wall -Wextra
BUILD_DIR ?= ../artifacts/build/runtime
UNAME_S := $(shell uname -s)
THREAD_FLAGS := -pthread
MATH_FLAGS := -lm
SOCKET_FLAGS :=
ifeq ($(OS),Windows_NT)
    SHARED_EXT := dll
    SHARED_FLAGS := -shared
    THREAD_FLAGS :=
    MATH_FLAGS :=
    SOCKET_FLAGS := -lws2_32
else ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    SHARED_FLAGS := -dynamiclib
else
    SHARED_EXT := so
    SHARED_FLAGS := -shared
endif

LIB_TQ := $(BUILD_DIR)/libturboquant.$(SHARED_EXT)
LIB_ASYNC := $(BUILD_DIR)/libnexa_async.$(SHARED_EXT)
LIB_Q4 := $(BUILD_DIR)/libnexa_q4.$(SHARED_EXT)
LIB_TRANSFORMER := $(BUILD_DIR)/libnexa_transformer.$(SHARED_EXT)
LIB_TQ_ATTENTION := $(BUILD_DIR)/libnexa_tq_attention.$(SHARED_EXT)
all: $(LIB_TQ) $(LIB_ASYNC) $(LIB_Q4) $(LIB_TRANSFORMER) $(LIB_TQ_ATTENTION)

$(BUILD_DIR):
	mkdir -p "$@"

$(LIB_TQ): turboquant.c turboquant.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(SHARED_FLAGS) -o "$@" turboquant.c $(MATH_FLAGS)

$(LIB_ASYNC): nexa_async.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o "$@" nexa_async.c $(SOCKET_FLAGS)

$(LIB_Q4): nexapack/q4.c nexapack/q4.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o "$@" nexapack/q4.c $(MATH_FLAGS)

$(LIB_TRANSFORMER): nexapack/transformer.c nexapack/transformer.h nexapack/q4.c nexapack/q4.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o "$@" nexapack/q4.c nexapack/transformer.c $(MATH_FLAGS)

$(LIB_TQ_ATTENTION): nexapack/tq_attention.c nexapack/tq_attention.h nexapack/q4.h turboquant.c turboquant.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(SHARED_FLAGS) -o "$@" turboquant.c nexapack/tq_attention.c $(MATH_FLAGS)

$(BUILD_DIR)/test_tq: turboquant_test.c turboquant.c turboquant.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) -o "$@" turboquant_test.c turboquant.c $(MATH_FLAGS)

$(BUILD_DIR)/test_tq_regressions: turboquant_regressions.c turboquant.c turboquant.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) -o "$@" turboquant_regressions.c turboquant.c $(MATH_FLAGS)

$(BUILD_DIR)/test_tq_mse_regressions: turboquant_mse_regressions.c turboquant.c turboquant.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) -o "$@" turboquant_mse_regressions.c $(MATH_FLAGS)

$(BUILD_DIR)/test_async_regressions: nexa_async_regressions.c nexa_async.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" nexa_async_regressions.c nexa_async.c $(SOCKET_FLAGS)

$(BUILD_DIR)/test_q4_regressions: nexapack/q4_regressions.c nexapack/q4.c nexapack/q4.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" nexapack/q4_regressions.c nexapack/q4.c $(MATH_FLAGS)

test: $(BUILD_DIR)/test_tq $(BUILD_DIR)/test_tq_regressions $(BUILD_DIR)/test_tq_mse_regressions $(BUILD_DIR)/test_async_regressions $(BUILD_DIR)/test_q4_regressions
	"$(BUILD_DIR)/test_tq"
	"$(BUILD_DIR)/test_tq_regressions"
	"$(BUILD_DIR)/test_tq_mse_regressions"
	"$(BUILD_DIR)/test_async_regressions"
	"$(BUILD_DIR)/test_q4_regressions"

clean:
	rm -f "$(LIB_TQ)" "$(LIB_ASYNC)" "$(LIB_Q4)" "$(LIB_TRANSFORMER)" "$(LIB_TQ_ATTENTION)" "$(BUILD_DIR)/test_tq" "$(BUILD_DIR)/test_tq_regressions" "$(BUILD_DIR)/test_tq_mse_regressions" "$(BUILD_DIR)/test_async_regressions" "$(BUILD_DIR)/test_q4_regressions"

.PHONY: all clean test
