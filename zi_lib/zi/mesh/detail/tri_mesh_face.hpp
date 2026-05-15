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

#ifndef ZI_MESH_TRI_MESH_FACE_HPP
#    define ZI_MESH_TRI_MESH_FACE_HPP 1

#    include <zi/bits/cstdint.hpp>
#    include <zi/bits/ref.hpp>
#    include <zi/bits/unordered_map.hpp>

#    include <zi/utility/assert.hpp>
#    include <zi/utility/enable_if.hpp>
#    include <zi/utility/non_copyable.hpp>
#    include <zi/utility/static_if.hpp>

#    include <cstddef>
#    include <iterator>

namespace zi
{
namespace mesh
{

// forward declaration
class tri_mesh;

namespace detail
{

struct tri_mesh_face_impl
{
private:
    uint32_t v_[3];

    static inline uint64_t make_edge(uint32_t x, uint32_t y)
    {
        return (static_cast<uint64_t>(~x) << 32) | (~y);
    }

public:
    void print() const { printf("%d %d %d\n", v_[0], v_[1], v_[2]); }

    inline bool operator==(const tri_mesh_face_impl& o) const
    {
        return std::equal(v_, v_ + 3, o.v_);
    }

    inline bool operator!=(const tri_mesh_face_impl& o) const
    {
        return !std::equal(v_, v_ + 3, o.v_);
    }

    template <std::size_t Index>
    inline uint32_t vertex(
        typename enable_if<(Index < 3), ::zi::detail::dummy<Index>>::type =
            0) const
    {
        return v_[Index];
    }

    inline uint32_t v0() const { return v_[0]; }
    inline uint32_t v1() const { return v_[1]; }
    inline uint32_t v2() const { return v_[2]; }

    inline uint64_t e0() const { return make_edge(v_[1], v_[2]); }
    inline uint64_t e1() const { return make_edge(v_[2], v_[0]); }
    inline uint64_t e2() const { return make_edge(v_[0], v_[1]); }

    inline uint64_t edge(std::size_t i)
    {
        ZI_ASSERT(i < 3);
        return make_edge(v_[i], v_[i == 2 ? 0 : i + 1]);
    }

    template <std::size_t Index>
    inline uint64_t edge(
        typename enable_if<(Index == 0), ::zi::detail::dummy<Index>>::type =
            0) const
    {
        return make_edge(v_[1], v_[2]);
    }

    template <std::size_t Index>
    inline uint64_t edge(
        typename enable_if<(Index == 1), ::zi::detail::dummy<Index>>::type =
            0) const
    {
        return make_edge(v_[2], v_[0]);
    }

    template <std::size_t Index>
    inline uint64_t edge(
        typename enable_if<(Index == 2), ::zi::detail::dummy<Index>>::type =
            0) const
    {
        return make_edge(v_[2], v_[0]);
    }

    inline uint64_t edge_from(uint32_t v) const
    {
        if (v == v_[0])
            return make_edge(v_[0], v_[1]);
        if (v == v_[1])
            return make_edge(v_[1], v_[2]);
        if (v == v_[2])
            return make_edge(v_[2], v_[0]);
        return 0;
    }

    inline tri_mesh_face_impl() {}

    inline tri_mesh_face_impl(uint32_t x, uint32_t y, uint32_t z)
    {
        v_[0] = x;
        v_[1] = y;
        v_[2] = z;
    }

    friend class tri_mesh;

private:
    inline void replace_vertex(uint32_t orig, uint32_t replacement)
    {
        if (orig == v_[0])
        {
            v_[0] = replacement;
            return;
        }

        if (orig == v_[1])
        {
            v_[1] = replacement;
            return;
        }

        if (orig == v_[2])
        {
            v_[2] = replacement;
            return;
        }

        ZI_VERIFY(0);
    }
};

struct face_slot
{
    tri_mesh_face_impl face;
    bool      alive;

    face_slot()
        : alive(false)
    {
    }
    face_slot(const tri_mesh_face_impl& face_)
        : face(face_)
        , alive(true)
    {
    }
};

struct trimesh_faces_iterator {
    const std::vector<face_slot>* faces;
    std::size_t idx;

    tri_mesh_face_impl operator*() const { return (*faces)[idx].face; }
    trimesh_faces_iterator& operator++() {
        do { 
            ++idx; 
        } while (idx < faces->size() && !(*faces)[idx].alive);
        return *this;
    }
        
    bool operator!=(const trimesh_faces_iterator& o) const { return idx != o.idx; }
};

struct trimesh_faces
{
private:
    std::vector<face_slot> faces_;
    std::vector<uint32_t>  free_list_;
    std::size_t            alive_count_;

public:
    trimesh_faces() { alive_count_ = 0; }

    void reserve(const std::size_t N) { faces_.reserve(N); }

    bool alive(const uint32_t id) const
    {
        if (id >= faces_.size())
        {
            return false;
        }
        return faces_[id].alive;
    }

    tri_mesh_face_impl get(const uint32_t id) const { 
        ZI_ASSERT(id < faces_.size());
        ZI_ASSERT(faces_[id].alive);
        return faces_[id].face;
    }

    std::size_t size() const { return alive_count_; }

    std::size_t push_back(const tri_mesh_face_impl& face)
    {
        alive_count_++;
        if (free_list_.empty())
        {
            faces_.push_back(face);
            return faces_.size() - 1;
        }
        else
        {
            uint32_t id      = free_list_.back();
            free_list_.pop_back();
            faces_[id].face  = face;
            faces_[id].alive = true;
            return id;
        }
    }

    void clear()
    {
        faces_.clear();
        free_list_.clear();
        alive_count_ = 0;
    }

    void erase(const uint32_t id)
    {
        alive_count_ -= static_cast<std::size_t>(faces_[id].alive);
        faces_[id].alive = false;
        free_list_.push_back(id);
    }

    trimesh_faces_iterator begin() const
    {
        auto it = trimesh_faces_iterator{&faces_, 0};
        if (!faces_.empty() && !faces_[0].alive) {
            ++it;
        }
        return it;
    }
    
    trimesh_faces_iterator end() const
    {
        return {&faces_, faces_.size()};
    }
};

} // namespace detail
} // namespace mesh
} // namespace zi

#endif
