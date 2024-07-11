#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/integral_like.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/colexicographical_lift.hpp"
#include "../../shapes/shape_size.hpp"
#include "../compose.hpp"
#include "../layouts/concepts/layout_like.hpp"
#include "../view_base.hpp"

namespace ubu
{

// lift's parens calls colexicographical_lift on a coordinate
template<coordinate To>
class lift
{
  public:
    constexpr lift(const To& c)
      : coshape_{c}
    {}

    lift(const lift&) = default;

    template<weakly_congruent<To> From>
    constexpr ubu::congruent<To> auto operator[](const From& coord) const
    {
      return colexicographical_lift(coord, coshape());
    }

    constexpr To coshape() const
    {
      return coshape_;
    }

  private:
    To coshape_;
};

// lifting_layout extends lift to present a layout interface
// also specializes compose
template<coordinate From, coordinate To>
  requires weakly_congruent<From,To>
class lifting_layout : public lift<To>, public view_base
{
  public:
    constexpr lifting_layout(const From& s, const To& c)
      : lift<To>(c), shape_{s}
    {}
  
    lifting_layout(const lifting_layout&) = default;
  
    constexpr From shape() const
    {
      return shape_;
    }

    constexpr integral_like auto size() const
    {
      return shape_size(shape());
    }
  
    template<layout_like_for<lifting_layout> L>
    constexpr layout_like auto compose(const L& rhs) const
    {
      // it's safe to discard shape_ when composing with another layout
      return ubu::compose(static_cast<const lift<To>&>(*this), rhs);
    }

  private:
    From shape_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

