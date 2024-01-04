#pragma once

#include "../../../detail/prologue.hpp"

#include "../compare/is_below.hpp"
#include "../concepts/coordinate.hpp"
#include "../zeros.hpp"


namespace ubu
{


// XXX this seems wrong, coord needs to be unwrapped when it's a single
template<scalar_coordinate C>
constexpr void lexicographical_increment(C& coord, const C&, const C&)
{
  ++coord;
}


template<nonscalar_coordinate C>
constexpr void lexicographical_increment(C& coord, const C& origin, const C& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C>
constexpr void lexicographical_increment_impl(C& coord, const C& origin, const C& end)
{
  // recurse into the Ith element
  lexicographical_increment(get<I>(coord), get<I>(origin), get<I>(end));

  // check the Ith element against the Ith bounds
  if(is_below(get<I>(coord), get<I>(end)))
  {
    return;
  }

  // roll over the Ith element to the origin
  if constexpr (I > 0)
  {
    // note that we don't roll the final (0th) dimension over to the origin
    get<I>(coord) = get<I>(origin);

    // continue recursion towards the left
    lexicographical_increment_impl<I-1>(coord, origin, end);
  }
}


} // end detail


template<nonscalar_coordinate C>
constexpr void lexicographical_increment(C& coord, const C& origin, const C& end)
{
  return detail::lexicographical_increment_impl<rank_v<C> - 1>(coord, origin, end);
}


template<coordinate C>
constexpr void lexicographical_increment(C& coord, const C& shape)
{
  return lexicographical_increment(coord, zeros<C>, shape);
}


} // end ubu


#include "../../../detail/epilogue.hpp"

