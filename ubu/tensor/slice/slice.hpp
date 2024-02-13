#pragma once

#include "../../detail/prologue.hpp"
#include "../tensor.hpp"
#include "slice_view.hpp"
#include "slicer.hpp"

namespace ubu
{
namespace detail
{

template<class T, class K>
concept has_slice_member_function = requires(T arg, K katana)
{
  { arg.slice(katana) } -> tensor_like;
};

template<class T, class K>
concept has_slice_free_function = requires(T arg, K katana)
{
  { slice(arg,katana) } -> tensor_like;
};

struct dispatch_slice
{
  template<class A, class K>
    requires has_slice_member_function<A&&,K&&>
  constexpr tensor_like auto operator()(A&& arg, K&& katana) const
  {
    return std::forward<A>(arg).slice(std::forward<K>(katana));
  }

  template<class A, class K>
    requires (not has_slice_member_function<A&&,K&&>
              and has_slice_free_function<A&&,K&&>)
  constexpr tensor_like auto operator()(A&& arg, K&& katana) const
  {
    return slice(std::forward<A>(arg), std::forward<K>(katana));
  }

  template<tensor_like A, slicer_for<tensor_shape_t<A>> K>
    requires (not has_slice_member_function<A&&,K&&>
              and not has_slice_free_function<A&&,K&&>)
  constexpr tensor_like auto operator()(A&& arg, K&& katana) const
  {
    return slice_view(std::forward<A>(arg), std::forward<K>(katana));
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_slice slice;

} // end anonymous namespace

} // end ubu

#include "../../detail/epilogue.hpp"

