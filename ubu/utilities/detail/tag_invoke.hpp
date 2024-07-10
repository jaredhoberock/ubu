#pragma once

#include "../../detail/prologue.hpp"
#include <utility>

namespace ubu::detail
{

// XXX this implementation of tag_invoke only looks for free functions

template<class CPO, class... Args>
concept has_tag_invoke_free_function = requires(CPO cpo, Args... args)
{
  tag_invoke(std::forward<CPO>(cpo), std::forward<Args>(args)...);
};

struct dispatch_tag_invoke
{
  template<class CPO, class... Args>
    requires has_tag_invoke_free_function<CPO&&,Args&&...>
  constexpr auto operator()(CPO&& cpo, Args&&... args) const
  {
    return tag_invoke(std::forward<CPO>(cpo), std::forward<Args>(args)...);
  }
};

inline constexpr detail::dispatch_tag_invoke tag_invoke;

template<class CPO, class... Args>
using tag_invoke_result_t = decltype(ubu::detail::tag_invoke(std::declval<CPO>(), std::declval<Args>()...));

template<class CPO, class... Args>
concept tag_invocable = requires(CPO cpo, Args... args)
{
  ubu::detail::tag_invoke(std::forward<CPO>(cpo), std::forward<Args>(args)...);
};

} // end ubu::detail

#include "../../detail/epilogue.hpp"

