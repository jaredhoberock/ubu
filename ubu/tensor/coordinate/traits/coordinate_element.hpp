#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include <tuple>

namespace ubu
{
namespace detail
{

template<std::size_t I, coordinate T>
struct coordinate_element
{
  using type = std::tuple_element_t<I,T>;
};

template<std::size_t I, scalar_coordinate T>
struct coordinate_element<I,T>
{
  using type = T;
};

} // end detail

template<std::size_t I, coordinate T>
using coordinate_element_t = typename detail::coordinate_element<I,T>::type;

} // end ubu

#include "../../../detail/epilogue.hpp"

