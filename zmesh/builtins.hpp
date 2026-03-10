#ifndef _ZMESH_BUILTINS_HXX_
#define _ZMESH_BUILTINS_HXX_


#ifdef _MSC_VER
#  include <intrin.h>
#  define popcount __popcnt

// https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
unsigned long ctz(unsigned long value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value)) {
        return trailing_zero;
    }
    else {
        return 32;
    }
}
#else
#  define popcount __builtin_popcount
#  define ctz __builtin_ctz
#endif

uint32_t ffs (uint32_t x) {
#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
   return __builtin_ffs(x);
#elif defined _MSC_VER
  /* _BitScanForward
     <https://docs.microsoft.com/en-us/cpp/intrinsics/bitscanforward-bitscanforward64> */
  unsigned long bit;
  if (_BitScanForward (&bit, x)) {
    return bit + 1;
  }
  return 0;
#else 
  if (x == 0) {
    return 0;
  }
  constexpr uint32_t num_bits = sizeof(x) * 8;
  for (uint32_t i = 0; i < num_bits; i++) {
    if ((x >> i) & 0x1) {
        return i + 1;
    }
  }
  return 0;
#endif
}

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