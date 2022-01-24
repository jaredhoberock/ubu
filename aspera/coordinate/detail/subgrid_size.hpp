#pragma once

#include "../../detail/prologue.hpp"

#include "../element.hpp"
#include "../grid_size.hpp"
#include <cstdint>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


constexpr std::size_t subgrid_size(const grid_coordinate auto& shape, std::index_sequence<>)
{
  return 1;
}


template<std::size_t axis0, std::size_t... axes>
constexpr std::size_t subgrid_size(const grid_coordinate auto& shape, std::index_sequence<axis0, axes...>)
{
  return ASPERA_NAMESPACE::grid_size(element<axis0>(shape)) * detail::subgrid_size(shape, std::index_sequence<axes...>{});
}


} // end detail

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

