#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../utilities/tuples.hpp"
#include "../../concepts/composable.hpp"
#include "../../concepts/nested_tensor_like.hpp"
#include "../../concepts/viewable_tensor_like.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/concepts/semicoordinate.hpp"
#include "../../coordinates/element.hpp"
#include "../../coordinates/coordinate_cast.hpp"
#include "../../coordinates/split_coordinate_at.hpp"
#include "../../coordinates/traits/rank.hpp"
#include "../../shapes/shape_size.hpp"
#include "../../traits/inner_tensor_shape.hpp"
#include "../all.hpp"
#include "../view_base.hpp"
#include "sliceable_with.hpp"
#include "slicer.hpp"
#include "unslice_coordinate.hpp"


namespace ubu
{
namespace detail
{


// slicing_layout and slice have a cyclic dependency and can't use each other directly
// declare detail::invoke_slice for slicing_layout's use
template<class... Args>
constexpr view auto invoke_slice(Args&&... args);


} // end detail


template<coordinate S, slicer K, coordinate R = unslice_coordinate_result_t<S,K>>
  requires congruent<R,unslice_coordinate_result_t<S,K>>
class slicing_layout : public view_base
{
  public:
    constexpr slicing_layout(S shape, K katana)
      : shape_(shape), katana_(katana)
    {}

    // this ctor is provided for CTAD. The coshape parameter is ignored otherwise.
    constexpr slicing_layout(S shape, K katana, R /*coshape*/)
      : slicing_layout(shape,katana)
    {}

    slicing_layout(const slicing_layout&) = default;

    constexpr S shape() const
    {
      return shape_;
    }
  
    template<congruent<S> C>
    constexpr R operator[](const C& coord) const
    {
      return coordinate_cast<R>(unslice_coordinate(coord, katana_));
    }

    constexpr auto size() const
    {
      return shape_size(shape());
    }

    // when K is underscore, returns all(tensor)
    template<tensor_like T, class K_ = K>
      requires detail::is_underscore_v<K_>
    friend constexpr view auto compose(T&& tensor, const slicing_layout& self)
    {
      return all(std::forward<T>(tensor));
    }

    // when
    // 1. T is nested, and
    // 2. katana's last dimension is a coordinate for T, and
    // 3. T's nested tensor is sliceable with katana's leading elements
    //
    // returns slice(element(tensor, last(katana)), leading(katana)))
    template<nested_tensor_like T>
      requires composable<T&&,slicing_layout>
               and coordinate_for<tuples::last_t<K>,T&&>
               and sliceable_with<tensor_reference_t<T&&>,tuples::first_t<detail::leading_and_last_result_t<K>>>
    friend constexpr view auto compose(T&& tensor, const slicing_layout& self)
    {
      auto [leading_katana, last_coord] = detail::leading_and_last(self.katana_);

      return detail::invoke_slice(element(std::forward<T>(tensor), last_coord), leading_katana);
    }

  private:
    S shape_;
    K katana_;
};


namespace detail
{

// slicing_layout and slice have a cyclic dependency and can't use each other directly
// define make_slicing_layout for slice's use as soon as slicing_layout is available
template<coordinate S, slicer K, coordinate R>
  requires congruent<R,unslice_coordinate_result_t<S,K>>
constexpr view auto make_slicing_layout(S shape, K katana, R coshape)
{
  return slicing_layout(shape, katana, coshape);
}

} // end detail

} // end ubu

#include "../../../detail/epilogue.hpp"

