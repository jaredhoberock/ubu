#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/weakly_congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/colexicographical_lift.hpp"
#include "../../shapes/shape_size.hpp"
#include "../view_base.hpp"

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

    constexpr std::size_t size() const
    {
      return shape_size(shape());
    }

    constexpr S coshape() const
    {
      return shape();
    }

  private:
    S shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

