#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/concepts/superdimensional.hpp"
#include "../../coordinates/coordinate_cast.hpp"
#include "../../coordinates/traits/default_coordinate.hpp"
#include "../../coordinates/one_extend_coordinate.hpp"
#include "../../coordinates/zero_extend_coordinate.hpp"
#include "../../shapes/shape_size.hpp"
#include "../view_base.hpp"

namespace ubu
{

// this layout appends zeros (to the right) of coord until it is congruent with To
// extending_layout is the opposite of truncating_layout
template<coordinate From, superdimensional<From> To, congruent<To> R = default_coordinate_t<To>>
  requires congruent<To,R>
class extending_layout : public view_base
{
  public:
    constexpr extending_layout(const From& shape)
      : shape_{shape}
    {}

    template<congruent<From> C>
    constexpr R operator[](const C& coord) const
    {
      return coordinate_cast<R>(zero_extend_coordinate<To>(coord));
    }

    constexpr From shape() const
    {
      return shape_;
    }

    constexpr congruent<To> auto coshape() const
    {
      return one_extend_coordinate<To>(shape());
    }

    constexpr std::size_t size() const
    {
      return shape_size(shape());
    }

  private:
    From shape_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

