#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool __nexa_resume(void*);
bool __nexa_is_done(void*);
void __nexa_destroy(void*);
int __nexa_listen_tcp(int, int);
int __nexa_socket_timeout(int, int);
int __nexa_socket_read(int, uint8_t*, int);
int __nexa_socket_write(int, const uint8_t*, int);
int __nexa_socket_close(int);

#ifndef _WIN32
#include <sys/socket.h>
static void socket_roundtrip(void) {
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(__nexa_socket_timeout(sockets[0], 1) == 0);
    assert(__nexa_socket_timeout(sockets[1], 1) == 0);
    const char message[] = "fragmented request";
    uint8_t received[sizeof(message)] = {0};
    assert(__nexa_socket_write(sockets[0], (const uint8_t*)message, sizeof(message)) == sizeof(message));
    int total = 0;
    while (total < (int)sizeof(message)) {
        int count = __nexa_socket_read(sockets[1], received + total, sizeof(message) - total);
        assert(count > 0);
        total += count;
    }
    assert(memcmp(message, received, sizeof(message)) == 0);
    assert(__nexa_socket_close(sockets[1]) == 0);
    /* A peer disconnect must not terminate the process with SIGPIPE. */
    assert(__nexa_socket_write(sockets[0], (const uint8_t*)message, sizeof(message)) == -1);
    assert(__nexa_socket_close(sockets[0]) == 0);
}
#endif

int main(void) {
    struct state { bool done; int32_t result; };
    struct state* s = malloc(sizeof(*s));
    s->done = true;
    s->result = 42;
    assert(__nexa_is_done(s));
    assert(!__nexa_resume(s));
    assert(s->result == 42);
    s->done = false;
    assert(!__nexa_is_done(s));
    assert(__nexa_resume(s));
    __nexa_destroy(s);
    __nexa_destroy(NULL);
    assert(__nexa_is_done(NULL));
    assert(__nexa_listen_tcp(-1, 1) == -1);
#ifndef _WIN32
    socket_roundtrip();
#endif
    puts("Async/socket runtime regressions passed");
    return 0;
}
