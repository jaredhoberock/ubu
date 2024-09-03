#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/integral_like.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/weakly_congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/colexicographical_lift.hpp"
#include "../../shapes/shape_size.hpp"
#include "../../traits/tensor_shape.hpp"
#include "../slices/slice_coordinate.hpp"
#include "../slices/slicer.hpp"
#include "../slices/slicing_layout.hpp"
#include "../all.hpp"
#include "../view_base.hpp"
#include "concepts/layout.hpp"
#include <utility>

namespace ubu
{

template<coordinate S>
class identity_layout : public view_base
{
  public:
    constexpr identity_layout(const S& shape)
      : shape_{shape}
    {}

    identity_layout(const identity_layout&) = default;

    template<weakly_congruent<S> C>
    constexpr congruent<S> auto operator[](const C& coord) const
    {
      return colexicographical_lift(coord, shape());
    }

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr integral_like auto size() const
    {
      return shape_size(shape());
    }

    constexpr S coshape() const
    {
      return shape();
    }

    template<tensor A>
      requires congruent<tensor_shape_t<A>,S>
    friend constexpr view auto compose(A&& a, const identity_layout&)
    {
      // XXX shouldn't this return trim(a, shape())?
      return all(std::forward<A>(a));
    }

    template<class B>
      requires composable<identity_layout,B&&>
    constexpr view auto compose(B&& b) const
    {
      // XXX the fact that we don't consider our shape and shape(b) is a little concerning
      return all(std::forward<B>(b));
    }

    template<slicer_for<S> K>
    constexpr layout auto slice(K katana) const
    {
      auto sliced_shape = slice_coordinate(shape(),katana);
      return slicing_layout(sliced_shape, katana, coshape());
    }

  private:
    S shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

