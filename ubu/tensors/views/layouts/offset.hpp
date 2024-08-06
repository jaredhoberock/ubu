#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/smaller.hpp"
#include "../../composed_tensor.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/coordinate_sum.hpp"
#include "../../traits/tensor_element.hpp"
#include "../../vectors/span_like.hpp"
#include "../all.hpp"
#include "concepts/layout_like.hpp"
#include "coshape.hpp"


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

} // end detail


template<layout_like L, coordinate O>
  requires view<L>
struct offset_layout : composed_view<detail::add_offset<O>, L>
{
  using super_t = composed_view<detail::add_offset<O>, L>;

  constexpr offset_layout(L l, O offset)
    : super_t(detail::add_offset<O>{offset}, l)
  {}

  template<class T>
    requires (rank_v<O> == 1)
  friend constexpr auto compose(T* ptr, const offset_layout& self)
  {
    return ubu::compose(ptr + self.a().offset, self.b());
  }

  template<span_like S>
    requires (rank_v<O> == 1)
  friend constexpr auto compose(S s, const offset_layout& self)
  {
    auto offset = self.a().offset;
    auto new_origin = smaller(offset, s.size());

    if constexpr(coshaped_layout_like<L>)
    {
      // if the layout has a coshape, we can use it to bound the size of the new span
      auto new_size = smaller(self.b().coshape(), s.size() - new_origin);

      return ubu::compose(s.subspan(new_origin, new_size), self.b());
    }
    else
    {
      return ubu::compose(s.subspan(new_origin), self.b());
    }
  }
};


template<layout_like L, congruent<tensor_element_t<L>> O>
constexpr view auto offset(L&& layout, O offset)
{
  return offset_layout(all(std::forward<L>(layout)), offset);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

