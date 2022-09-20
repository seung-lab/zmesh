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

#include <cstdint>
#include <type_traits>

namespace zi::mesh
{

namespace detail
{

template <class T>
struct identity_type
{
    using type = T;
};

template <class T>
using identity_type_t = typename identity_type<T>::type;

template <class UnderlyingT, unsigned XBits, unsigned YBits, unsigned ZBits>
struct packed_coordinate_helper
{
private:
    static_assert(std::is_unsigned_v<UnderlyingT>);
    static_assert(sizeof(UnderlyingT) * 8 >= XBits + YBits + ZBits);
    static_assert(XBits > 0 && YBits > 0 && ZBits > 0);

    static constexpr UnderlyingT one = 1;

public:
    using underlying_type = UnderlyingT;

    static constexpr int z_bitsize = ZBits;
    static constexpr int y_bitsize = YBits;
    static constexpr int x_bitsize = XBits;

    static constexpr int z_shift = 0;
    static constexpr int y_shift = z_bitsize;
    static constexpr int x_shift = z_bitsize + y_bitsize;

    static constexpr underlying_type z_unit = one << z_shift;
    static constexpr underlying_type y_unit = one << y_shift;
    static constexpr underlying_type x_unit = one << x_shift;

    static constexpr underlying_type z_endmask = (one << z_bitsize) - 1;
    static constexpr underlying_type y_endmask = (one << y_bitsize) - 1;
    static constexpr underlying_type x_endmask = (one << x_bitsize) - 1;

    static constexpr underlying_type z_mask = z_endmask << z_shift;
    static constexpr underlying_type y_mask = y_endmask << y_shift;
    static constexpr underlying_type x_mask = z_endmask << x_shift;

public:
    static constexpr underlying_type extract_x(UnderlyingT v) noexcept
    {
        return (v >> x_shift); // & x_endmask;
    }

    static constexpr underlying_type extract_y(UnderlyingT v) noexcept
    {
        return (v >> y_shift) & y_endmask;
    }

    static constexpr underlying_type extract_z(UnderlyingT v) noexcept
    {
        return (v >> z_shift) & z_endmask;
    }

    template <class F>
    static constexpr F extract_x_as(UnderlyingT v) noexcept
    {
        return static_cast<F>(extract_x(v));
    }

    template <class F>
    static constexpr F extract_y_as(UnderlyingT v) noexcept
    {
        return static_cast<F>(extract_y(v));
    }

    template <class F>
    static constexpr F extract_z_as(UnderlyingT v) noexcept
    {
        return static_cast<F>(extract_z(v));
    }

    template <class F>
    static constexpr F
    extract_transformed_x_as(UnderlyingT v, identity_type_t<F> shift = 0,
                             identity_type_t<F> scale = 1) noexcept
    {
        return scale * (shift + extract_x_as<F>(v));
    }

    template <class F>
    static constexpr F
    extract_transformed_y_as(UnderlyingT v, identity_type_t<F> shift = 0,
                             identity_type_t<F> scale = 1) noexcept
    {
        return scale * (shift + extract_y_as<F>(v));
    }

    template <class F>
    static constexpr F
    extract_transformed_z_as(UnderlyingT v, identity_type_t<F> shift = 0,
                             identity_type_t<F> scale = 1) noexcept
    {
        return scale * (shift + extract_z_as<F>(v));
    }
};

} // namespace detail

template <class UnderlyingT>
struct packed_coordinate;

template <>
struct packed_coordinate<std::uint32_t>
    : detail::packed_coordinate_helper<std::uint64_t, 11, 11, 10>
{
};

template <>
struct packed_coordinate<std::uint64_t>
    : detail::packed_coordinate_helper<std::uint64_t, 21, 21, 21>
{
};

} // namespace zi::mesh
