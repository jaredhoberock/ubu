#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/math/coordinate_sum.hpp"
#include "../compose.hpp"
#include "../traits/tensor_element.hpp"
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

  constexpr O offset() const
  {
    return super_t::tensor().offset;
  }

  template<class T>
    requires (rank_v<O> == 1)
  friend auto compose(T* ptr, const offset_layout& self)
  {
    return ubu::compose(ptr + self.tensor().offset, self.layout());
  }

  template<class T>
    requires (rank_v<O> == 1)
  friend auto compose(const std::span<T>& s, const offset_layout& self)
  {
    auto offset = self.tensor().offset;
    std::size_t new_origin = offset <= s.size() ? offset : s.size();
    return ubu::compose(s.subspan(new_origin), self.layout());
  }
};


template<layout L, congruent<tensor_element_t<L>> O>
constexpr auto offset(L layout, O offset)
{
  return offset_layout(layout, offset);
}


} // end ubu

#include "../../detail/epilogue.hpp"

