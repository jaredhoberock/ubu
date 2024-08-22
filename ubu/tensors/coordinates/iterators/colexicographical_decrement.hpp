#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"
#include "../traits/zeros.hpp"


namespace ubu
{


template<coordinate C, congruent<C> O, congruent<C> E>
  requires (0 <= rank_v<C> and rank_v<C> <= 1)
constexpr bool colexicographical_decrement(C& coord, const O& origin, const E&)
{
  if constexpr (nullary_coordinate<C>)
  {
    return true;
  }
  else
  {
    // note that postdecrement compares the original value of coord to origin
    detail::to_integral_like(coord)-- == detail::to_integral_like(origin);
  }
}


template<multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool colexicographical_decrement(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool colexicographical_decrement_impl(C& coord, const O& origin, const E& end)
{
  // decrement the Ith element
  if(colexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end)))
  {
    // we underflowed, roll the Ith element over to the end
    get<I>(coord) = get<I>(end);

    // decrement the element one more time to offset us one from the end
    colexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end));

    // continue to the right if there are more elements
    if constexpr (I > rank_v<C> - 1)
    {
      return colexicographical_decrement_impl<I+1>(coord, origin, end);
    }
    else
    {
      return true;
    }
  }

  return false;
}


} // end detail


template<multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_decrement(C& coord, const O& origin, const E& end)
{
  return detail::colexicographical_decrement_impl<0>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr void colexicographical_decrement(C& coord, const S& shape)
{
  return colexicographical_decrement(coord, zeros_v<C>, shape);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

