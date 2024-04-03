#pragma once

#include "../../../detail/prologue.hpp"

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

// note that the type of coordinate_element_v<I,T> is different from coordinate_element_t<I,T>
// because, for example, this definition "unwraps" a constant<13> to 13
// XXX is this a good choice?
template<std::size_t I, semicoordinate T>
  requires (I < rank_v<T>) and constant_valued<coordinate_element_t<I,T>>
constexpr inline auto coordinate_element_v = constant_value_v<coordinate_element_t<I,T>>;

} // end ubu

#include "../../../detail/epilogue.hpp"

