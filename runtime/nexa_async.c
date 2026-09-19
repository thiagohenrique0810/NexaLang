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
#include <stdio.h>
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
__declspec(dllexport) int sched_yield(void) { Sleep(0); return 0; }
#else
#include <sched.h>
#endif

#ifdef _WIN32
#define NEXA_API __declspec(dllexport)
#else
#define NEXA_API
#endif

NEXA_API void* __nexa_stderr(void) { return stderr; }

/* Resume a coroutine.
 * In the current eager-evaluation model, async functions run to completion
 * immediately when called, so resume is a cooperative yield. */
NEXA_API bool __nexa_resume(void *handle) {
    if (!handle) return false;
    /* The done flag is the first byte of the state struct */
    volatile uint8_t *done = (volatile uint8_t *)handle;
    if (*done) return false;
    /* Yield to let other work proceed */
    sched_yield();
    return true;
}

/* Check if a coroutine has completed. */
NEXA_API bool __nexa_is_done(void *handle) {
    if (!handle) return true;
    uint8_t *done = (uint8_t *)handle;
    return *done != 0;
}

/* Destroy and free a coroutine state. */
NEXA_API void __nexa_destroy(void *handle) {
    if (handle) {
        free(handle);
    }
}

/* Small POSIX socket bridge: platform-specific sockaddr and option constants
 * belong in C, not in NexaLang's portable standard-library source. */
#include <errno.h>
#include <limits.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#else
#define close closesocket
#endif

NEXA_API int __nexa_socket_close(int fd) { return close(fd); }

NEXA_API int __nexa_socket_accept(int fd) {
#ifdef _WIN32
    SOCKET client = accept((SOCKET)fd, NULL, NULL);
    if (client == INVALID_SOCKET) return -1;
    if (client > INT_MAX) { closesocket(client); return -1; }
    return (int)client;
#else
    int client;
    do { client = accept(fd, NULL, NULL); } while (client < 0 && errno == EINTR);
    return client;
#endif
}

NEXA_API int __nexa_socket_read(int fd, uint8_t* data, int length) {
    if (!data || length < 0) return -1;
#ifdef _WIN32
    return recv((SOCKET)fd, (char*)data, length, 0);
#else
    ssize_t n;
    do { n = recv(fd, data, (size_t)length, 0); } while (n < 0 && errno == EINTR);
    return (int)n;
#endif
}

NEXA_API int __nexa_listen_tcp(int port, int backlog) {
    if (port < 0 || port > 65535 || backlog <= 0) return -1;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) return -1;
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) return -1;
    if (sock > INT_MAX) { closesocket(sock); return -1; }
    int fd = (int)sock;
#else
    int fd = socket(AF_INET, SOCK_STREAM, 0);
#endif
    if (fd < 0) return -1;
    int enabled = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&enabled, sizeof(enabled));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
#ifdef __APPLE__
    addr.sin_len = sizeof(addr);
#endif
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, (const struct sockaddr*)&addr, sizeof(addr)) || listen(fd, backlog)) {
        close(fd);
        return -1;
    }
    return fd;
}

NEXA_API int __nexa_socket_timeout(int fd, int seconds) {
    if (seconds < 0) return -1;
#ifdef _WIN32
    DWORD timeout = seconds > INT_MAX / 1000 ? INT_MAX : (DWORD)seconds * 1000;
#else
    struct timeval timeout = {seconds, 0};
#endif
    int rc = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout))) rc = -1;
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, (const char*)&enabled, sizeof(enabled))) rc = -1;
#endif
    return rc;
}

NEXA_API int __nexa_socket_write(int fd, const uint8_t* data, int length) {
    if (!data || length < 0) return -1;
    int sent = 0;
    while (sent < length) {
#ifdef _WIN32
        int n = send((SOCKET)fd, (const char*)data + sent, length - sent, 0);
        if (n == SOCKET_ERROR && WSAGetLastError() == WSAEINTR) continue;
#elif defined(MSG_NOSIGNAL)
        ssize_t n = send(fd, data + sent, (size_t)(length - sent), MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, data + sent, (size_t)(length - sent), 0);
#endif
#ifndef _WIN32
        if (n < 0 && errno == EINTR) continue;
#endif
        if (n <= 0) return -1;
        sent += (int)n;
    }
    return sent;
}

/* HTTP field names use ASCII case folding, independent of locale/platform. */
NEXA_API int __nexa_ascii_ncasecmp(const uint8_t* a, const uint8_t* b, int length) {
    if (!a || !b || length < 0) return -1;
    for (int i = 0; i < length; i++) {
        unsigned int ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca += 'a' - 'A';
        if (cb >= 'A' && cb <= 'Z') cb += 'a' - 'A';
        if (ca != cb) return ca < cb ? -1 : 1;
        if (!ca) return 0;
    }
    return 0;
}
