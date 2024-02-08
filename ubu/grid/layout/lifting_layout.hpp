#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/colexicographical_lift.hpp"
#include "../compose.hpp"

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
    constexpr To operator[](const From& coord) const
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
class lifting_layout : public lift<To>
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
  
    template<layout_for<lifting_layout> L>
    constexpr layout auto compose(const L& rhs) const
    {
      // it's safe to discard shape_ when composing with another layout
      return ubu::compose(static_cast<const lift<To>&>(*this), rhs);
    }

  private:
    From shape_;
};

} // end ubu

#include "../../detail/epilogue.hpp"

