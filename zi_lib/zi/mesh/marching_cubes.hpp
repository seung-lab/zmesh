//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

// Algorithm adopted from http://paulbourke.net/geometry/polygonise/
// and modified for the particular use case.

// Extended with uint32_t support in 2019 by William Silversmith

#pragma once

#include "detail/all_equal.hpp"
#include "detail/mc_tables.hpp"

#include <zi/mesh/network_sort.hpp>
#include <zi/vl/vec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace zi::mesh
{

template <typename T>
struct mc_masks;

template <>
struct mc_masks<std::uint64_t>
{
    static const std::uint64_t zshift = 0;  // z = 21 bits (2^20 - 1) + .5
    static const std::uint64_t yshift = 21; // y = 21 bits (2^20 - 1) + .5
    static const std::uint64_t xshift = 42; // x = 21 bits (2^20 - 1) + .5

    static const std::uint64_t z_mask = 0x1FFFFF;
    static const std::uint64_t y_mask = z_mask << yshift;
    static const std::uint64_t x_mask = z_mask << xshift;

    static const std::uint64_t unit_z = 1;
    static const std::uint64_t unit_y = unit_z << yshift;
    static const std::uint64_t unit_x = unit_z << xshift;
};

template <>
struct mc_masks<std::uint32_t>
{
    static const std::uint64_t zshift = 0;  // z = 10 bits (511.5)
    static const std::uint64_t yshift = 10; // y = 11 bits (1023.5)
    static const std::uint64_t xshift =
        21; // x = 11 bits (1023.5) (in theory...)

    static const std::uint64_t z_mask = 0x3FF;
    static const std::uint64_t y_mask = 0x7FF << yshift;
    static const std::uint64_t x_mask = 0x7FF << xshift;

    static const std::uint64_t unit_z = 1;
    static const std::uint64_t unit_y = unit_z << yshift;
    static const std::uint64_t unit_x = unit_z << xshift;
};

template <typename PositionType, typename LabelType>
class marching_cubes
{
private:
    static_assert(std::is_same_v<PositionType, std::uint64_t> ||
                      std::is_same_v<PositionType, std::uint32_t>,
                  "Unsupported PositionType");

private:
    marching_cubes(marching_cubes const&) = delete;
    marching_cubes& operator=(marching_cubes const&) = delete;

    // static std::size_t const tri_table_end = 0xffffffff;
    // static std::size_t const edge_table[256];
    // static std::size_t const tri_table[256][16];

    using mask_traits = mc_masks<PositionType>;

public:
    static constexpr inline PositionType
    pack_coords(PositionType x, PositionType y, PositionType z) noexcept
    {
        return (x << mask_traits::xshift) | (y << mask_traits::yshift) | z;
    }

    template <class T>
    static inline T unpack_x(PositionType packed, const T& offset = T(0),
                             const T& factor = T(1))
    {
        return factor * (offset + ((packed & mask_traits::x_mask) >>
                                   mask_traits::xshift));
    }

    template <class T>
    static inline T unpack_y(PositionType packed, const T& offset = T(0),
                             const T& factor = T(1))
    {
        return factor * (offset + ((packed & mask_traits::y_mask) >>
                                   mask_traits::yshift));
    }

    template <class T>
    static inline T unpack_z(PositionType packed, const T& offset = T(0),
                             const T& factor = T(1))
    {
        return factor * (offset + (packed & mask_traits::z_mask));
    }

public:
    struct packed_printer
    {
    private:
        const PositionType coor_;

        packed_printer(packed_printer const&) = delete;
        packed_printer* operator=(packed_printer const&) = delete;

    public:
        packed_printer(const PositionType c)
            : coor_(c)
        {
        }

        inline friend std::ostream& operator<<(std::ostream&         os,
                                               const packed_printer& p)
        {
            os << "[ "
               << marching_cubes<PositionType, LabelType>::template unpack_x<
                      PositionType>(p.coor_)
               << ", "
               << marching_cubes<PositionType, LabelType>::template unpack_y<
                      PositionType>(p.coor_)
               << ", "
               << marching_cubes<PositionType, LabelType>::template unpack_z<
                      PositionType>(p.coor_)
               << " ]";
            return os;
        }
    };

public:
    using triangle_t = vl::vec<PositionType, 3>;

    std::size_t                                            num_faces_;
    std::unordered_map<LabelType, std::vector<triangle_t>> meshes_;

public:
    const std::unordered_map<LabelType, std::vector<triangle_t>>& meshes() const
    {
        return meshes_;
    }

    marching_cubes()
        : num_faces_(0)
        , meshes_()
    {
    }

    void clear()
    {
        meshes_.clear();
        meshes_.rehash(0);
        num_faces_ = 0;
    }

    bool erase(const LabelType& t)
    {
        try
        {
            std::size_t face_delta = meshes_.at(t).size();
            std::size_t num_erased = meshes_.erase(t);
            num_faces_ -= face_delta;
            return num_erased > 0;
        }
        catch (const std::out_of_range& oor)
        {
            return false;
        }
    }

    std::size_t face_count() const { return num_faces_; }

    std::size_t count(const LabelType& t) const { return meshes_.count(t); }

    std::size_t size() const { return meshes_.size(); }

    static constexpr inline PositionType midpoint(PositionType p1,
                                                  PositionType p2) noexcept
    {
        return (p1 >> 1) + (p2 >> 1);
    }

private:
    struct c_order_tag
    {
    };
    struct fortran_order_tag
    {
    };

    template <class Fn, class Tag>
    static inline void mc_nested_loops(
        std::size_t sx, std::size_t sy, std::size_t sz, 
        Fn&& fn, Tag const&
    ) {
        if constexpr (std::is_same_v<Tag, c_order_tag>)
        {
            for (std::size_t x = 0; x < sx - 1; ++x)
            {
                for (std::size_t y = 0; y < sy - 1; ++y)
                {
                    for (std::size_t z = 0; z < sz - 1; ++z)
                    {
                        fn(x, y, z, z + sz * (y + sy * x));
                    }
                }
            }
        }
        else
        {
            static_assert(std::is_same_v<Tag, fortran_order_tag>);
            for (std::size_t z = 0; z < sz - 1; ++z)
            {
                for (std::size_t y = 0; y < sy - 1; ++y)
                {
                    for (std::size_t x = 0; x < sx - 1; ++x)
                    {
                        fn(x, y, z, x + sx * (y + sy * z));
                    }
                }
            }
        }
    }

    template <class Tag>
    static inline std::array<std::size_t, 7>
    get_strides(std::size_t sx, std::size_t sy, std::size_t sz, Tag const&)
    {
        if constexpr (std::is_same_v<Tag, c_order_tag>)
        {
            return {
                static_cast<std::size_t>(sy * sz),          // +x
                static_cast<std::size_t>(sy * sz + 1),      // +x +z
                static_cast<std::size_t>(1),                // +z
                static_cast<std::size_t>(sz),               // +y
                static_cast<std::size_t>(sy * sz + sz),     // +x +y
                static_cast<std::size_t>(sy * sz + sz + 1), // +x +y +z
                static_cast<std::size_t>(sz + 1)            // +y +z
            };
        }
        else
        {
            static_assert(std::is_same_v<Tag, fortran_order_tag>);

            return {
                static_cast<std::size_t>(1),                // +x
                static_cast<std::size_t>(1 + sx * sy),      // +x +z
                static_cast<std::size_t>(sx * sy),          // +z
                static_cast<std::size_t>(sx),               // +y
                static_cast<std::size_t>(1 + sx),           // +x +y
                static_cast<std::size_t>(1 + sx + sx * sy), // +x +y +z
                static_cast<std::size_t>(sx + sx * sy)      // +y +z
            };
        }
    }

    template <class Tag>
    void marche(const LabelType* data, std::size_t const sx,
                std::size_t const sy, std::size_t const sz,
                Tag const& order_tag)
    {
        constexpr std::array<PositionType, 8> cube_corners = {
            pack_coords(0, 0, 0), pack_coords(2, 0, 0), pack_coords(2, 0, 2),
            pack_coords(0, 0, 2), pack_coords(0, 2, 0), pack_coords(2, 2, 0),
            pack_coords(2, 2, 2), pack_coords(0, 2, 2)};

        constexpr std::array<PositionType, 12> edge_midpoints = {
            midpoint(cube_corners[0], cube_corners[1]),
            midpoint(cube_corners[1], cube_corners[2]),
            midpoint(cube_corners[2], cube_corners[3]),
            midpoint(cube_corners[3], cube_corners[0]),
            midpoint(cube_corners[4], cube_corners[5]),
            midpoint(cube_corners[5], cube_corners[6]),
            midpoint(cube_corners[6], cube_corners[7]),
            midpoint(cube_corners[7], cube_corners[4]),
            midpoint(cube_corners[0], cube_corners[4]),
            midpoint(cube_corners[1], cube_corners[5]),
            midpoint(cube_corners[2], cube_corners[6]),
            midpoint(cube_corners[3], cube_corners[7])};

        auto strides = get_strides(sx, sy, sz, order_tag);

        static_network_sorter<8> network_sort;

        mc_nested_loops(
            sx, sy, sz,
            [&](std::size_t x, std::size_t y, std::size_t z, std::size_t ind)
            {
                std::array<LabelType, 8> const labels = {
                    data[ind],
                    data[ind + strides[0]],
                    data[ind + strides[1]],
                    data[ind + strides[2]],
                    data[ind + strides[3]],
                    data[ind + strides[4]],
                    data[ind + strides[5]],
                    data[ind + strides[6]]};

                if (all_equal(labels))
                {
                    return;
                }

                // Instead of using std::unordered_set or similar
                // to get unique labels, use a high efficiency sort,
                // a "network sort", for a fixed size labels and then
                // iterate from high to low values and skip repeats.
                // This Saves almost 40% of the march time. We make an
                // array copy before sorting to preserve the structure
                // in labels. std::unordered_set uses a hash with closed
                // addressing + chaining which is inefficient for our
                // case.
                std::array<LabelType, 8> ulabels = labels;
                network_sort(ulabels);

                for (int i = 7; i >= 0; i--)
                {
                    const LabelType label = ulabels[i];
                    if (label == 0)
                    {
                        break;
                    }
                    else if (i < 7 && ulabels[i + 1] == label)
                    {
                        continue;
                    }

                    std::size_t c = 0;

                    for (std::size_t n = 0; n < 8; ++n)
                    {
                        c |= (labels[n] != label) << n;
                    }

                    if (!mc_edge_table[c])
                    {
                        continue;
                    }

                    PositionType cur =
                        ((x * mask_traits::unit_x) | (y * mask_traits::unit_y) |
                         (z * mask_traits::unit_z))
                        << 1;

                    for (std::size_t n = 0;
                         mc_triangle_table[c][n] != mc_triangle_table_end;
                         n += 3)
                    {
                        ++num_faces_;
                        meshes_[label].emplace_back(
                            edge_midpoints[mc_triangle_table[c][n + 2]] + cur,
                            edge_midpoints[mc_triangle_table[c][n + 1]] + cur,
                            edge_midpoints[mc_triangle_table[c][n]] + cur);
                    }
                }
            },
            order_tag);
    }

public:
    void marche(const LabelType* data, std::size_t const sx,
                std::size_t const sy, std::size_t const sz,
                const bool c_order = true)
    {
        if (c_order)
        {
            marche(data, sx, sy, sz, c_order_tag());
        }
        else
        {
            marche(data, sx, sy, sz, fortran_order_tag());
        }
    }

    std::vector<triangle_t> const& get_triangles(LabelType const& id) const
    {
        return meshes_.find(id)->second;
    }
};

} // namespace zi::mesh
