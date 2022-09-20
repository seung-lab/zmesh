// The MIT License
//
// Copyright 2022 Aleksandar Zlateski <aleksandar.zlateski@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <array>
#include <cstdint>
#include <type_traits>

namespace zi::mesh
{

namespace detail
{

template <class T, std::size_t N, std::size_t... Elems>
bool all_equal_helper(std::array<T, N> const& arr,
                      std::index_sequence<Elems...> const&) noexcept
{
    static_assert(sizeof...(Elems) + 1 == N);
    return (... && (arr[0] == arr[Elems + 1]));
}

} // namespace detail

template <class T, std::size_t N>
bool all_equal(std::array<T, N> const& arr) noexcept
{
    if constexpr (N < 2)
    {
        return true;
    }
    else
    {
        return detail::all_equal_helper(arr, std::make_index_sequence<N - 1>());
    }
}

} // namespace zi::mesh
