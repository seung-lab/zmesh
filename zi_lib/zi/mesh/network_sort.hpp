#pragma once

#include <array>
#include <utility>

namespace zi::mesh {

// optimal network sort for 8 elements
// https://bertdobbelaere.github.io/sorting_networks.html#CCFS16
// [(0,2),(1,3),(4,6),(5,7)]
// [(0,4),(1,5),(2,6),(3,7)]
// [(0,1),(2,3),(4,5),(6,7)]
// [(2,4),(3,5)]
// [(1,4),(3,6)]
// [(1,2),(3,4),(5,6)]

#define CMP_SWAP(x,y) \
    if (arr[x] > arr[y]) {\
        std::swap(arr[x], arr[y]);\
    }

template <typename T>
void sort_8(std::array<T, 8>& arr) {
    CMP_SWAP(0,2)
    CMP_SWAP(1,3)
    CMP_SWAP(4,6)
    CMP_SWAP(5,7)
    CMP_SWAP(0,4)
    CMP_SWAP(1,5)
    CMP_SWAP(2,6)
    CMP_SWAP(3,7)
    CMP_SWAP(0,1)
    CMP_SWAP(2,3)
    CMP_SWAP(4,5)
    CMP_SWAP(6,7)
    CMP_SWAP(2,4)
    CMP_SWAP(3,5)
    CMP_SWAP(1,4)
    CMP_SWAP(3,6)
    CMP_SWAP(1,2)
    CMP_SWAP(3,4)
    CMP_SWAP(5,6)
}

#undef CMP_SWAP

};