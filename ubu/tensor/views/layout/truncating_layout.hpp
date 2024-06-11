#pragma once

#include "../../../detail/prologue.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/concepts/subdimensional.hpp"
#include "../../coordinates/coordinate_cast.hpp"
#include "../../coordinates/traits/default_coordinate.hpp"
#include "../../coordinates/one_extend_coordinate.hpp"
#include "../../coordinates/truncate_coordinate.hpp"
#include "../../shapes/shape_size.hpp"
#include <ranges>

namespace ubu
{


// this layout simply drops modes (from the right) of coord until
// it is congruent with To
template<coordinate From, subdimensional<From> To, congruent<To> R = default_coordinate_t<To>>
  requires congruent<To,R>
class truncating_layout : public std::ranges::view_base
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

#include "../../../detail/epilogue.hpp"

