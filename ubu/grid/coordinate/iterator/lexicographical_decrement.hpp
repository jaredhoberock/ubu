#pragma once

#include "../../../detail/prologue.hpp"

#include "../compare/is_below.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/as_integral.hpp"
#include "../zeros.hpp"


namespace ubu
{


template<scalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void lexicographical_decrement(C& coord, const O&, const E&)
{
  --detail::as_integral(coord);
}


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void lexicographical_decrement(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void lexicographical_decrement_impl(C& coord, const O& origin, const E& end)
{
  // is the Ith element of coord at the origin?
  if(is_below_or_equal(get<I>(coord), get<I>(origin)))
  {
    // set the Ith element to the end
    get<I>(coord) = get<I>(end);

    // decrement the element one more time to offset us one from the end
    lexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end));

    if constexpr (I > 0)
    {
      // continue recursion towards the left
      lexicographical_decrement_impl<I-1>(coord, origin, end);
    }
  }
  else
  {
    lexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end));
  }
}


} // end detail


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void lexicographical_decrement(C& coord, const O& origin, const E& end)
{
  return detail::lexicographical_decrement_impl<rank_v<C> - 1>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr void lexicographical_decrement(C& coord, const S& shape)
{
  return lexicographical_decrement(coord, zeros<C>, shape);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

