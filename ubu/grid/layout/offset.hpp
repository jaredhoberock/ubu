#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/coordinate_sum.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../compose.hpp"
#include "../grid.hpp"
#include "../view.hpp"
#include "layout.hpp"
#include <span>


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


template<layout L, coordinate O>
struct offset_layout : view<detail::add_offset<O>, L>
{
  using super_t = view<detail::add_offset<O>, L>;

  constexpr offset_layout(L l, O offset)
    : super_t(detail::add_offset<O>{offset}, l)
  {}

  template<class T>
    requires (rank_v<O> == 1)
  friend auto compose(T* ptr, const offset_layout& self)
  {
    return ubu::compose(ptr + self.grid().offset, self.layout());
  }

  template<class T>
    requires (rank_v<O> == 1)
  friend auto compose(const std::span<T>& s, const offset_layout& self)
  {
    auto offset = self.grid().offset;
    std::size_t new_origin = offset <= s.size() ? offset : s.size();
    return ubu::compose(s.subspan(new_origin), self.layout());
  }
};


template<layout L, congruent<grid_element_t<L>> O>
constexpr auto offset(L layout, O offset)
{
  return offset_layout(layout, offset);
}


} // end ubu

#include "../../detail/epilogue.hpp"
