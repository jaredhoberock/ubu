#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/integrals/bounded_integral_like.hpp"
#include "../detail/to_integral_like.hpp"
#include "../detail/tuple_algorithm.hpp"

namespace ubu
{
namespace detail
{

template<coordinate C>
constexpr bool elements_are_bounded()
{
  if constexpr(scalar_coordinate<C>)
  {
    return bounded_integral_like<to_integral_like_t<C>>;
  }
  else
  {
    using T = std::remove_cvref_t<C>;

    // XXX this assumes that T is default-constructible
    T dummy_tuple{};

    return tuple_all(dummy_tuple, [](auto e)
    {
      return elements_are_bounded<decltype(e)>();
    });
  }
}


} // end detail


template<class T>
concept bounded_coordinate =
  coordinate<T>
  and detail::elements_are_bounded<T>()
;


} // end ubu

#include "../../../detail/epilogue.hpp"

