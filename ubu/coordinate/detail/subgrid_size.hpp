#pragma once

#include "../../detail/prologue.hpp"

#include "../element.hpp"
#include "../grid_size.hpp"
#include <cstdint>
#include <utility>


namespace ubu::detail
{


constexpr std::size_t subgrid_size(const grid_coordinate auto& shape, std::index_sequence<>)
{
  return 1;
}


template<std::size_t axis0, std::size_t... axes>
constexpr std::size_t subgrid_size(const grid_coordinate auto& shape, std::index_sequence<axis0, axes...>)
{
  return ubu::grid_size(element<axis0>(shape)) * detail::subgrid_size(shape, std::index_sequence<axes...>{});
}


} // end ubu::detail


#include "../../detail/epilogue.hpp"

