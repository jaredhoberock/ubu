#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../views/all.hpp"
#include "uniform_masked_view.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class M>
concept has_mask_member_function = requires(T t, M m)
{
  { t.mask(m) } -> view;
};

template<class T, class M>
concept has_mask_free_function = requires(T t, M m)
{
  { mask(t,m) } -> view;
};

struct dispatch_mask
{
  template<class T, class M>
    requires has_mask_member_function<T&&,M&&>
  constexpr view auto operator()(T&& t, M&& m) const
  {
    return std::forward<T>(t).mask(std::forward<M>(m));
  }

  template<class T, class M>
    requires (not has_mask_member_function<T&&,M&&>
              and has_mask_free_function<T&&,M&&>)
  constexpr view auto operator()(T&& t, M&& m) const
  {
    return mask(std::forward<T>(t), std::forward<M>(m));
  }

  template<tensor_like T>
    requires (not has_mask_member_function<T&&,bool>
              and not has_mask_free_function<T&&,bool>)
  constexpr view auto operator()(T&& t, bool m) const
  {
    return uniform_masked_view(all(std::forward<T>(t)), m);
  }
};

} // end detail

constexpr inline detail::dispatch_mask mask;

} // end ubu

#include "../../detail/epilogue.hpp"

