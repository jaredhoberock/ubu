#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integrals/integral_like.hpp"
#include "../../../../miscellaneous/tuples.hpp"
#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/weakly_congruent.hpp"
#include "../../../coordinates/coordinate_sum.hpp"
#include "../../../coordinates/detail/to_integral_like.hpp"
#include "apply_stride.hpp"


namespace ubu
{
namespace detail
{


template<integral_like R, scalar_coordinate D, scalar_coordinate C>
constexpr R apply_stride_r_impl(const D& stride, const C& coord)
{
  return static_cast<R>(to_integral_like(stride)) * static_cast<R>(to_integral_like(coord));
}


template<coordinate R, nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr auto apply_stride_r_impl(const D& stride, const C& coord);


template<nonscalar_coordinate R, nonscalar_coordinate D, scalar_coordinate C>
constexpr R apply_stride_r_impl(const D& stride, const C& coord)
{
  return tuples::zip_with(stride, R{}, [&](const auto& s, auto r)
  {
    return apply_stride_r_impl<decltype(r)>(s, coord);
  });
}


template<coordinate R, nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr auto apply_stride_r_impl(const D& stride, const C& coord)
{
  auto star = [](const auto& s, const auto& c)
  {
    return apply_stride_r_impl<R>(s,c);
  };

  auto plus = [](const auto& c1, const auto& c2)
  {
    return coordinate_sum(c1,c2);
  };

  return tuples::inner_product(stride, coord, star, plus);
}


} // end detail


// this variant of apply_stride allows control over the width of the result type, R
// R must be congruent to the result returned by the other variant, apply_stride()
template<coordinate R, coordinate D, weakly_congruent<D> C>
  requires (congruent<R, apply_stride_t<D,C>>
            and not std::is_reference_v<R>)
constexpr R apply_stride_r(const D& stride, const C& coord)
{
  return detail::apply_stride_r_impl<R>(stride, coord);
}


} // end ubu

#include "../../../../detail/epilogue.hpp"

