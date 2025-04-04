#pragma once

#include "../../../detail/prologue.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "concepts/layout.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_coshape_member_function = requires(T layout)
{
  { layout.coshape() } -> coordinate;
};

template<class T>
concept has_coshape_free_function = requires(T layout)
{
  { coshape(layout) } -> coordinate;
};


struct dispatch_coshape
{
  template<class L>
    requires has_coshape_member_function<L&&>
  constexpr coordinate auto operator()(L&& layout) const
  {
    return std::forward<L&&>(layout).coshape();
  }

  template<class L>
    requires (not has_coshape_member_function<L&&>
              and has_coshape_free_function<L&&>)
  constexpr coordinate auto operator()(L&& layout) const
  {
    return coshape(std::forward<L&&>(layout));
  }
};


} // end detail

constexpr inline detail::dispatch_coshape coshape;

template<class T>
using coshape_t = decltype(coshape(std::declval<T>()));

template<class T>
concept coshaped =
  requires(T arg)
  {
    coshape(arg);
  }
;

} // end ubu

#include "../../../detail/epilogue.hpp"

