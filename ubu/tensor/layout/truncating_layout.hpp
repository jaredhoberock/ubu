#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/concepts/subdimensional.hpp"
#include "../coordinate/coordinate_cast.hpp"
#include "../coordinate/traits/default_coordinate.hpp"
#include "../coordinate/one_extend_coordinate.hpp"
#include "../coordinate/truncate_coordinate.hpp"
#include "../shape/shape_size.hpp"

namespace ubu
{


// this layout simply drops modes (from the right) of coord until
// it is congruent with To
template<coordinate From, subdimensional<From> To, congruent<To> R = default_coordinate_t<To>>
  requires congruent<To,R>
class truncating_layout
{
  public:
    constexpr truncating_layout(const To& coshape)
      : coshape_{coshape}
    {}

    template<congruent<From> C>
    constexpr R operator[](const C& coord) const
    {
      return coordinate_cast<R>(truncate_coordinate<To>(coord));
    }

    constexpr congruent<From> auto shape() const
    {
      return one_extend_coordinate<From>(coshape());
    }

    constexpr To coshape() const
    {
      return coshape_;
    }

    constexpr std::size_t size() const
    {
      // note that for this layout, this is equivalent to shape_size(shape())
      return shape_size(coshape());
    }

  private:
    To coshape_;
};


} // end ubu

#include "../../detail/epilogue.hpp"

