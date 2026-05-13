#ifndef _ZMESH_BUILTINS_HXX_
#define _ZMESH_BUILTINS_HXX_

#ifdef _MSC_VER
#  include <intrin.h>
#  define zmesh_popcount __popcnt
// Source - https://stackoverflow.com/a/20468180
// Posted by crazyjul
// Retrieved 2026-05-12, License - CC BY-SA 3.0
unsigned long __inline zmesh_ctz (unsigned long value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value)) {
        return trailing_zero;
    }
    else {
        return 32; // undefined if value 0, choose 32 as a sensible choice
    }
}
unsigned long __inline zmesh_clz (unsigned long value) {
    unsigned long leading_zero = 0;

    if (_BitScanReverse(&leading_zero, value)) {
       return 31 - leading_zero;
    }
    else {
         return 32; // undefined if value 0, choose 32 as a sensible choice
    }
}
#else
#  define zmesh_popcount __builtin_popcount
#  define zmesh_ctz __builtin_ctz
#  define zmesh_clz __builtin_clz
#endif

// Windows implementation of getline thanks to claude.ai
#ifdef _WIN32
#include <cstdio>
#define SSIZE_MAX ((int64_t)(SIZE_MAX / 2))
static int64_t getline(char** __restrict lineptr, size_t* __restrict n, FILE* __restrict stream) {
    if (lineptr == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    if (*lineptr == NULL || *n == 0) {
        *n = 128;
        *lineptr = (char*)malloc(*n);
        if (!*lineptr) return -1;
    }

    int64_t pos = 0;
    int c;

    while ((c = fgetc(stream)) != EOF) {
        if (pos + 1 >= *n) {
            if (*n > SSIZE_MAX / 2) {
                errno = EOVERFLOW;
                return -1;
            }
            *n *= 2;
            char* tmp = (char*)realloc(*lineptr, *n);
            if (!tmp) return -1;
            *lineptr = tmp;
        }
        (*lineptr)[pos++] = c;
        if (c == '\n') break;
    }

    if (pos == 0) return -1;
    (*lineptr)[pos] = '\0';
    return (int64_t)pos;
}
#endif

#endif