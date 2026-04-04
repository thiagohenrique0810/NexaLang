/* NexaLang Async Runtime — Cooperative coroutine helpers
 *
 * Async functions in NexaLang use a manual state struct:
 *   struct { bool done; T result; }
 *
 * The coroutine handle is an opaque i8* pointer to this state.
 * These functions provide the C-level runtime support for:
 *   - Resuming a coroutine (currently a no-op since state is eagerly evaluated)
 *   - Checking completion status
 *   - Destroying/freeing the coroutine state
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>

/* Resume a coroutine.
 * In the current eager-evaluation model, async functions run to completion
 * immediately when called, so resume is a cooperative yield. */
bool __nexa_resume(void *handle) {
    if (!handle) return false;
    /* The done flag is the first byte of the state struct */
    volatile uint8_t *done = (volatile uint8_t *)handle;
    if (*done) return false;
    /* Yield to let other work proceed */
    sched_yield();
    return true;
}

/* Check if a coroutine has completed. */
bool __nexa_is_done(void *handle) {
    if (!handle) return true;
    uint8_t *done = (uint8_t *)handle;
    return *done != 0;
}

/* Destroy and free a coroutine state. */
void __nexa_destroy(void *handle) {
    if (handle) {
        free(handle);
    }
}
