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

// Extended with uint32_t support in 2019 by William Silversmith

#ifndef ZI_MESH_MARCHING_CUBES_HPP
#define ZI_MESH_MARCHING_CUBES_HPP 1

#include <zi/mesh/tri_mesh.hpp>
#include <zi/mesh/quadratic_simplifier.hpp>

#include <zi/bits/cstdint.hpp>
#include <zi/bits/type_traits.hpp>
#include <zi/bits/unordered_map.hpp>
#include <zi/bits/unordered_set.hpp>

#include <zi/utility/non_copyable.hpp>

#include <zi/vl/vec.hpp>

#include <cstddef>
#include <vector>
#include <ostream>

// Logic: If two adjacent labels match, return their avg position. 
//        Else, return the first vertex position.
// 
// The code used to read: (vert[p1]+vert[p2]) >> 1 
// However, this can cause overflow in uint32 vertex types
// when the carry bit reaches from Z to Y or Y to X. 
//
// Each vertex has a bit at the beginning representing 2^-1
// which reads 0 initially. By shifting down first before adding
// we ensure that the fields never overflow.

#define ZI_MC_QUICK_INTERP( p1, p2, val )                      \
    (((( vals[ p1 ] == val ) ^ ( vals[ p2 ] == val )) ?        \
     ((vert[ p1 ] >> 1) + (vert[ p2 ] >> 1)) : vert[ p1 ] ))


namespace zi {
namespace mesh {

template <typename T> 
struct mc_masks {
    static const std::size_t zshift = 0;   // z = 21 bits (2^20 - 1) + .5
    static const std::size_t yshift = 21;  // y = 21 bits (2^20 - 1) + .5
    static const std::size_t xshift = 42;  // x = 21 bits (2^20 - 1) + .5

    static const std::size_t z_mask = 0x1FFFFF;
    static const std::size_t y_mask = z_mask << yshift;
    static const std::size_t x_mask = z_mask << xshift;

    static const std::size_t xy_mask = x_mask | y_mask;
    static const std::size_t xz_mask = x_mask | z_mask;
    static const std::size_t yz_mask = y_mask | z_mask;

    static const std::size_t delta_z = 1;
    static const std::size_t delta_y = delta_z << yshift;
    static const std::size_t delta_x = delta_z << xshift;

    static const std::size_t delta_2z = 2;
    static const std::size_t delta_2y = delta_2z << yshift;
    static const std::size_t delta_2x = delta_2z << xshift;
};

template <> 
struct mc_masks<uint32_t> {
    static const std::size_t zshift = 0;   // z = 10 bits (511.5)
    static const std::size_t yshift = 10;  // y = 11 bits (1023.5)
    static const std::size_t xshift = 21;  // x = 11 bits (1023.5) (in theory...)

    static const std::size_t z_mask = 0x3FF;
    static const std::size_t y_mask = 0x7FF << yshift;
    static const std::size_t x_mask = 0x7FF << xshift;

    static const std::size_t xy_mask = x_mask | y_mask;
    static const std::size_t xz_mask = x_mask | z_mask;
    static const std::size_t yz_mask = y_mask | z_mask;

    static const std::size_t delta_z = 1;
    static const std::size_t delta_y = delta_z << yshift;
    static const std::size_t delta_x = delta_z << xshift;

    static const std::size_t delta_2z = 2;
    static const std::size_t delta_2y = delta_2z << yshift;
    static const std::size_t delta_2x = delta_2z << xshift;  
};

template< typename PositionType, typename LabelType >
class marching_cubes: non_copyable
{
private:
    typedef marching_cubes< PositionType, LabelType > this_type;

    ZI_STATIC_ASSERT( is_integral< PositionType >::value, non_integral_type_for_marching_cubes );

    static const std::size_t tri_table_end = 0xffffffff;
    static const std::size_t edge_table[ 256 ];
    static const std::size_t tri_table[ 256 ][ 16 ];
    static const mc_masks<PositionType> masks;

public:
    static inline PositionType pack_coords( PositionType x, PositionType y, PositionType z )
    {
        return ( x << masks.xshift ) | ( y << masks.yshift ) | z;
    }

    template< class T >
    static inline T unpack_x( PositionType packed, const T& offset = T( 0 ), const T& factor = T( 1 ) )
    {
        return factor * ( offset + ((packed & masks.x_mask) >> masks.xshift) );
    }

    template< class T >
    static inline T unpack_y( PositionType packed, const T& offset = T( 0 ), const T& factor = T( 1 ) )
    {
        return factor * ( offset + ((packed & masks.y_mask) >> masks.yshift) );
    }

    template< class T >
    static inline T unpack_z( PositionType packed, const T& offset = T( 0 ), const T& factor = T( 1 ) )
    {
        return factor * ( offset + ( packed & masks.z_mask ) );
    }    

public:
    struct packed_printer: non_copyable
    {
    private:
        const PositionType coor_;

    public:
        packed_printer( const PositionType c ): coor_( c )
        {
        }

        inline friend
        std::ostream& operator<<( std::ostream& os, const packed_printer& p )
        {
            os << "[ "
               << marching_cubes< PositionType, LabelType >::template unpack_x< PositionType >( p.coor_ ) << ", "
               << marching_cubes< PositionType, LabelType >::template unpack_y< PositionType >( p.coor_ ) << ", "
               << marching_cubes< PositionType, LabelType >::template unpack_z< PositionType >( p.coor_ ) << " ]";
            return os;
        }
    };


public:

    typedef vl::vec<PositionType, 3>                    triangle;
    std::size_t                                         num_faces_;
    unordered_map< LabelType, std::vector< triangle > > meshes_   ;

public:

    const unordered_map< LabelType, std::vector< triangle > >& meshes() const
    {
        return meshes_;
    }

    marching_cubes()
        : num_faces_( 0 ),
          meshes_()
    {
    }

    void clear()
    {
        meshes_.clear();
        num_faces_ = 0;
    }

    bool erase(const LabelType& t) 
    {
        try {
            std::size_t face_delta = meshes_.at(t).size();
            std::size_t num_erased = meshes_.erase(t);
            num_faces_ -= face_delta;
            return num_erased > 0;
        }
        catch (const std::out_of_range& oor) {
            return false;
        }
    }

    std::size_t face_count() const
    {
        return num_faces_;
    }

    std::size_t count( const LabelType& t ) const
    {
        return meshes_.count( t );
    }

    std::size_t size() const
    {
        return meshes_.size();
    }

    void marche( const LabelType* data, std::size_t x_dim, std::size_t y_dim, std::size_t z_dim )
    {

        unordered_set< LabelType > local;

        // If we don't use uint64_t, then uint32_t
        // messes up in the final position due to some
        // kind of truncation or overflow issue. We use
        // all the bits in a uint32_t, but only 63 bits
        // for the uint64_t, so we don't see the issue
        // either way.
        uint64_t ptrs_[ 12 ];

        PositionType vert[ 8 ] =
        {
            pack_coords( 0, 0, 0 ),
            pack_coords( 2, 0, 0 ),
            pack_coords( 2, 0, 2 ),
            pack_coords( 0, 0, 2 ),
            pack_coords( 0, 2, 0 ),
            pack_coords( 2, 2, 0 ),
            pack_coords( 2, 2, 2 ),
            pack_coords( 0, 2, 2 )
        };

        const std::size_t off1 = y_dim * z_dim;
        const std::size_t off2 = y_dim * z_dim + 1;
        const std::size_t off3 = 1;
        const std::size_t off4 = z_dim;
        const std::size_t off5 = y_dim * z_dim + z_dim;
        const std::size_t off6 = y_dim * z_dim + z_dim + 1;
        const std::size_t off7 = z_dim + 1;

        const std::size_t x_max = x_dim - 1;
        const std::size_t y_max = y_dim - 1;
        const std::size_t z_max = z_dim - 1;

        const std::size_t yz_dim = y_dim * z_dim;

        std::size_t x_off = 0;
        std::size_t y_off = 0;

        LabelType label = 0;

        for ( std::size_t x = 0; x < x_max; ++x )
        {
            for ( std::size_t y = 0; y < y_max; ++y )
            {
                for ( std::size_t z = 0; z < z_max; ++z )
                {
                    const std::size_t ind = x_off + y_off + z;

                    const LabelType vals[ 8 ] =
                        {
                            data[ ind ],
                            data[ ind + off1 ],
                            data[ ind + off2 ],
                            data[ ind + off3 ],
                            data[ ind + off4 ],
                            data[ ind + off5 ],
                            data[ ind + off6 ],
                            data[ ind + off7 ]
                        };

                    local.clear();

                    for ( std::size_t i = 0; i < 8; ++i )
                    {
                        if ( vals[ i ] )
                        {
                            local.insert( vals[ i ] );
                        }
                    }

                    FOR_EACH( it, local )
                    {
                        std::size_t c = 0;

                        for ( std::size_t n = 0; n < 8; ++n )
                        {
                            if ( vals[ n ] != *it )
                            {
                                c |= ( 1 << n );
                            }
                        }

                        if ( edge_table[ c ] )
                        {
                            label = *it;

                            if ( edge_table[ c ] & 1   ) ptrs_[  0 ] = ZI_MC_QUICK_INTERP( 0, 1, label );
                            if ( edge_table[ c ] & 2   ) ptrs_[  1 ] = ZI_MC_QUICK_INTERP( 1, 2, label );
                            if ( edge_table[ c ] & 4   ) ptrs_[  2 ] = ZI_MC_QUICK_INTERP( 2, 3, label );
                            if ( edge_table[ c ] & 8   ) ptrs_[  3 ] = ZI_MC_QUICK_INTERP( 3, 0, label );
                            if ( edge_table[ c ] & 16  ) ptrs_[  4 ] = ZI_MC_QUICK_INTERP( 4, 5, label );
                            if ( edge_table[ c ] & 32  ) ptrs_[  5 ] = ZI_MC_QUICK_INTERP( 5, 6, label );
                            if ( edge_table[ c ] & 64  ) ptrs_[  6 ] = ZI_MC_QUICK_INTERP( 6, 7, label );
                            if ( edge_table[ c ] & 128 ) ptrs_[  7 ] = ZI_MC_QUICK_INTERP( 7, 4, label );
                            if ( edge_table[ c ] & 256 ) ptrs_[  8 ] = ZI_MC_QUICK_INTERP( 0, 4, label );
                            if ( edge_table[ c ] & 512 ) ptrs_[  9 ] = ZI_MC_QUICK_INTERP( 1, 5, label );
                            if ( edge_table[ c ] & 1024) ptrs_[ 10 ] = ZI_MC_QUICK_INTERP( 2, 6, label );
                            if ( edge_table[ c ] & 2048) ptrs_[ 11 ] = ZI_MC_QUICK_INTERP( 3, 7, label );

                            for ( std::size_t n = 0; tri_table[ c ][ n ] != tri_table_end; n += 3)
                            {
                                ++num_faces_;
                                meshes_[ *it ].push_back
                                    ( triangle( ptrs_[ tri_table[ c ][ n + 2 ] ],
                                                ptrs_[ tri_table[ c ][ n + 1 ] ],
                                                ptrs_[ tri_table[ c ][ n ] ] ) );
                            }

                        }

                    }

                    // move 2 units z
                    for ( std::size_t i = 0; i < 8 ; ++i )
                    {
                        vert[ i ] += 2;
                    }
                }

                // move 2 units y
                for ( std::size_t i = 0; i < 8 ; ++i )
                {
                    vert[ i ] += masks.delta_2y;
                }

                // set every other vertex to z = 0, 2
                for ( std::size_t i = 0; i < 8 ; ++i )
                {   
                    vert[ i ]  = ( vert[ i ] & masks.xy_mask ) | ( i & 2 );
                }

                y_off += z_dim;
            }

            // move 2 units x
            for ( std::size_t i = 0; i < 8 ; ++i )
            {
                vert[ i ] += masks.delta_2x;
            }

            // set y to to 0,0,0,0, 2,2,2,2
            for ( std::size_t i = 0; i < 8 ; ++i )
            {
                vert[ i ]  = ( vert[ i ] & masks.xz_mask ) | ( ( i & 4 ) << (masks.yshift - 1) );
            }

            y_off = 0;
            x_off += yz_dim;
        }

    }

    template< class T > std::size_t
    fill_tri_mesh( const LabelType& id,
                   tri_mesh& ret,
                   std::vector< vl::vec< T, 3 > >& points,
                   const T& xtrans = T( 0 ),
                   const T& ytrans = T( 0 ),
                   const T& ztrans = T( 0 ),
                   const T& xscale = T( 1 ),
                   const T& yscale = T( 1 ),
                   const T& zscale = T( 1 ) ) const
    {
        if ( !meshes_.count( id ) )
        {
            return 0;
        }

        uint32_t idx = 0;
        unordered_map< PositionType, uint32_t > pts;

        const std::vector< triangle >& data = meshes_.find( id )->second;

        FOR_EACH( it, data )
        {
            if ( !pts.count( it->at(0) ) )
            {
                pts.insert( std::make_pair( it->at(0), idx++ ) );
            }
            if ( !pts.count( it->at(1) ) )
            {
                pts.insert( std::make_pair( it->at(1), idx++ ) );
            }
            if ( !pts.count( it->at(2) ) )
            {
                pts.insert( std::make_pair( it->at(2), idx++ ) );
            }
        }

        ret.resize( idx );
        points.resize( idx );

        FOR_EACH( it, pts )
        {
            points[ it->second ] = vl::vec< T, 3 >
                ( this_type::template unpack_x< T >( it->first, xtrans, xscale ),
                  this_type::template unpack_y< T >( it->first, ytrans, yscale ),
                  this_type::template unpack_z< T >( it->first, ztrans, zscale ) );


        }

        FOR_EACH( it, data )
        {
            ret.add_face( pts[ it->at(0) ], pts[ it->at(1) ], pts[ it->at(2) ] );
        }

        return idx;

    }

    const std::vector< triangle >& get_triangles( const LabelType& id ) const
    {
        return meshes_.find(id)->second;
    }

    template< class T > std::size_t
    fill_simplifier( ::zi::mesh::simplifier< T >& ret,
                     const LabelType& id,
                     const T& xtrans = T( 0 ),
                     const T& ytrans = T( 0 ),
                     const T& ztrans = T( 0 ),
                     const T& xscale = T( 1 ),
                     const T& yscale = T( 1 ),
                     const T& zscale = T( 1 ) ) const
    {
        if ( !meshes_.count( id ) )
        {
            return 0;
        }

        uint32_t idx = 0;
        unordered_map< PositionType, uint32_t > pts;

        const std::vector< triangle >& data = meshes_.find( id )->second;

        FOR_EACH( it, data )
        {
            if ( !pts.count( it->at(0) ) )
            {
                pts.insert( std::make_pair( it->at(0), idx++ ) );
            }
            if ( !pts.count( it->at(1) ) )
            {
                pts.insert( std::make_pair( it->at(1), idx++ ) );
            }
            if ( !pts.count( it->at(2) ) )
            {
                pts.insert( std::make_pair( it->at(2), idx++ ) );
            }
        }

        ret.resize( idx );

        FOR_EACH( it, pts )
        {
            ret.point( it->second ) = vl::vec< T, 3 >
                ( this_type::template unpack_x< T >( it->first, xtrans, xscale ),
                  this_type::template unpack_y< T >( it->first, ytrans, yscale ),
                  this_type::template unpack_z< T >( it->first, ztrans, zscale ) );


        }

        FOR_EACH( it, data )
        {
            ret.add_face( pts[ it->at(0) ], pts[ it->at(1) ], pts[ it->at(2) ] );
        }

        return idx;

    }


};

#  define ZI_MESH_MARCHING_CUBES_HPP_INLUDING_TABLES 1
#  define ZI_MESH_MARCHING_CUBES_TYPE( var, ext )                       \
    template< typename PositionType, typename LabelType > const std::size_t marching_cubes< PositionType, LabelType >::var ext
#
#  include <zi/mesh/detail/marching_cubes_tables.hpp>
#
#  undef ZI_MESH_MARCHING_CUBES_TYPE
#  undef ZI_MESH_MARCHING_CUBES_HPP_INLUDING_TABLES

#undef ZI_MC_QUICK_INTERP

} // namespace mesh
} // namespace zi

#endif
