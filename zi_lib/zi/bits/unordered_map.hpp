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

#ifndef ZI_BITS_UNORDERED_MAP_HPP
#define ZI_BITS_UNORDERED_MAP_HPP 1

#include <zi/config/config.hpp>
#include <zi/bits/hash.hpp>

#include <unordered_map>
#define ZI_UNORDERED_MAP_NAMESPACE ::std

namespace zi {

using ZI_UNORDERED_MAP_NAMESPACE::unordered_map;
using ZI_UNORDERED_MAP_NAMESPACE::unordered_multimap;
using ZI_UNORDERED_MAP_NAMESPACE::swap;

} // namespace zi

#undef ZI_UNORDERED_MAP_NAMESPACE
#endif
