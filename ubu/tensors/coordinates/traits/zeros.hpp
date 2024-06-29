#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/constant.hpp"
#include "../../../miscellaneous/constant_valued.hpp"
#include "../detail/tuple_algorithm.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include <limits>
#include <type_traits>


namespace ubu
{
namespace detail
{


template<integral_like I>
constexpr integral_like auto integral_like_to_zero()
{
  using T = std::remove_cvref_t<I>;

  // first check if T has a constant value that happens to be zero
  if constexpr (constant_valued<T>)
  {
    if constexpr (T{} == 0)
    {
      return T{};
    }
    else
    {
      // XXX if we're going to preserve constant, we might also wish to preserve T::value_type, if possible
      return constant<0>{};
    }
  }

  // failing that, check if 0 is in the representable range of T
  else if constexpr(std::numeric_limits<T>::min() <= 0 and 0 <= std::numeric_limits<T>::max())
  {
    return T{0};
  }

  // failing that, give up on T and just return an int 0
  else
  {
    return int(0);
  }
}


template<coordinate C>
constexpr congruent<C> auto zeros()
{
  if constexpr (integral_like<C>)
  {
    return integral_like_to_zero<C>();
  }
  else
  {
    using T = std::remove_cvref_t<C>;

    // XXX this assumes that T is default-constructible
    T dummy_tuple{};

    return tuple_zip_with(dummy_tuple, [](auto e)
    {
      return zeros<decltype(e)>();
    });
  }
}

} // end detail


template<coordinate C>
constexpr congruent<C> auto zeros_v = detail::zeros<C>();

template<coordinate C>
using zeros_t = decltype(zeros_v<C>);

} // end ubu

#include "../../../detail/epilogue.hpp"

