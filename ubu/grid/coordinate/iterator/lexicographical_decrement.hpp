#pragma once

#include "../../../detail/prologue.hpp"

#include "../compare/is_below.hpp"
#include "../concepts/coordinate.hpp"
#include "../element.hpp"
#include "../zeros.hpp"


namespace ubu
{


template<scalar_coordinate C>
constexpr void lexicographical_decrement(C& coord, const C&, const C&)
{
  --coord;
}


template<nonscalar_coordinate C>
constexpr void lexicographical_decrement(C& coord, const C& origin, const C& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C>
constexpr void lexicographical_decrement_impl(C& coord, const C& origin, const C& end)
{
  // is the Ith element of coord at the origin?
  if(is_below_or_equal(element<I>(coord), element<I>(origin)))
  {
    // set the Ith element to the end
    element<I>(coord) = element<I>(end);

    // decrement the element one more time to offset us one from the end
    lexicographical_decrement(element<I>(coord), element<I>(origin), element<I>(end));

    if constexpr (I > 0)
    {
      // continue recursion towards the left
      lexicographical_decrement_impl<I-1>(coord, origin, end);
    }
  }
  else
  {
    lexicographical_decrement(element<I>(coord), element<I>(origin), element<I>(end));
  }
}


} // end detail


template<nonscalar_coordinate C>
constexpr void lexicographical_decrement(C& coord, const C& origin, const C& end)
{
  return detail::lexicographical_decrement_impl<rank_v<C> - 1>(coord, origin, end);
}


template<coordinate C>
constexpr void lexicographical_decrement(C& coord, const C& shape)
{
  return lexicographical_decrement(coord, zeros<C>, shape);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

