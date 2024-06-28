#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/constant_valued.hpp"
#include "../concepts/semicoordinate.hpp"
#include "../traits/rank.hpp"
#include <tuple>

namespace ubu
{
namespace detail
{

template<std::size_t I, semicoordinate T>
struct coordinate_element
{
  using type = std::tuple_element_t<I,T>;
};

template<std::size_t I, semicoordinate T>
  requires (I == 0 and rank_v<T> == 1)
struct coordinate_element<I,T>
{
  using type = T;
};

} // end detail


template<std::size_t I, semicoordinate T>
  requires (I < rank_v<T>)
using coordinate_element_t = typename detail::coordinate_element<I,T>::type;


template<std::size_t I, semicoordinate T>
  requires (I < rank_v<T>) and constant_valued<coordinate_element_t<I,T>>
constexpr inline coordinate_element_t<I,T> coordinate_element_v;


} // end ubu

#include "../../../detail/epilogue.hpp"

