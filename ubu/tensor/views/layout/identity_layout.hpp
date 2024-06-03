#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/coordinate.hpp"
#include "../../coordinate/colexicographical_lift.hpp"
#include <ranges>

namespace ubu
{

template<coordinate S>
class identity_layout : std::ranges::view_base
{
  public:
    constexpr identity_layout(const S& shape)
      : shape_{shape}
    {}

    identity_layout(const identity_layout&) = default;

    template<weakly_congruent<S> C>
    constexpr S operator[](const C& coord)
    {
      return colexicographical_lift(coord, shape());
    }

    constexpr S shape() const
    {
      return shape_;
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

