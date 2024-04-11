#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/smaller.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/math/coordinate_sum.hpp"
#include "../compose.hpp"
#include "../traits/tensor_element.hpp"
#include "../view.hpp"
#include "../vector/span_like.hpp"
#include "layout.hpp"


namespace ubu
{
namespace detail
{

template<coordinate O>
struct add_offset
{
  O offset;

  template<congruent<O> C>
  constexpr congruent<O> auto operator()(const C& coord) const
  {
    return coordinate_sum(offset, coord);
  }
};

// XXX coshape() needs to be a CPO
template<class T>
concept has_coshape = requires(T layout)
{
  { layout.coshape() } -> coordinate;
};

} // end detail


template<layout L, coordinate O>
struct offset_layout : view<detail::add_offset<O>, L>
{
  using super_t = view<detail::add_offset<O>, L>;

  constexpr offset_layout(L l, O offset)
    : super_t(detail::add_offset<O>{offset}, l)
  {}

  constexpr O offset() const
  {
    return super_t::tensor().offset;
  }

  template<class T>
    requires (rank_v<O> == 1)
  friend constexpr auto compose(T* ptr, const offset_layout& self)
  {
    return ubu::compose(ptr + self.tensor().offset, self.layout());
  }

  template<span_like S>
    requires (rank_v<O> == 1)
  friend constexpr auto compose(S s, const offset_layout& self)
  {
    auto offset = self.tensor().offset;
    auto new_origin = smaller(offset, s.size());

    if constexpr(detail::has_coshape<L>)
    {
      // if the layout has a coshape, we can use it to bound the size of the new span
      auto new_size = smaller(self.layout().coshape(), s.size() - new_origin);

      return ubu::compose(s.subspan(new_origin, new_size), self.layout());
    }
    else
    {
      return ubu::compose(s.subspan(new_origin), self.layout());
    }
  }
};


template<layout L, congruent<tensor_element_t<L>> O>
constexpr auto offset(L layout, O offset)
{
  return offset_layout(layout, offset);
}


} // end ubu

#include "../../detail/epilogue.hpp"

