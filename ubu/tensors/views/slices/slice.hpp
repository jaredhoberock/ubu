#pragma once

#include "../../../detail/prologue.hpp"
#include "../../concepts/tensor_like.hpp"
#include "../../concepts/view.hpp"
#include "../all.hpp"
#include "sliced_view.hpp"
#include "slicer.hpp"
#include "underscore.hpp"

namespace ubu
{
namespace detail
{

template<class T, class K>
concept has_slice_member_function = requires(T arg, K katana)
{
  { arg.slice(katana) } -> view;
};

template<class T, class K>
concept has_slice_free_function = requires(T arg, K katana)
{
  { slice(arg,katana) } -> view;
};

// a slicer without underscores yields
// a single element of T
template<class S, class T>
concept singular_slicer_for =
  tensor_like<T>
  and congruent<tensor_shape_t<T>,S>
  and coordinate<S>
;

struct dispatch_slice
{
  template<class A, class K>
    requires has_slice_member_function<A&&,K&&>
  constexpr view auto operator()(A&& arg, K&& katana) const
  {
    return std::forward<A>(arg).slice(std::forward<K>(katana));
  }

  template<class A, class K>
    requires (not has_slice_member_function<A&&,K&&>
              and has_slice_free_function<A&&,K&&>)
  constexpr view auto operator()(A&& arg, K&& katana) const
  {
    return slice(std::forward<A>(arg), std::forward<K>(katana));
  }

  template<tensor_like A, slicer_for<tensor_shape_t<A>> K>
    requires (not has_slice_member_function<A&&,K>
              and not has_slice_free_function<A&&,K>)
  constexpr view auto operator()(A&& arg, K katana) const
  {
    if constexpr (is_underscore_v<K>)
    {
      // when katana is _, we simply return all of arg
      return all(std::forward<A>(arg));
    }
    else if constexpr (singular_slicer_for<K,A>)
    {
      static_assert(not singular_slicer_for<K,A>, "slice: Singular slice is currently unsupported.");
      return all(std::forward<A>(arg));
    }
    else
    {
      // otherwise, return a sliced_view of all of arg
      return sliced_view(all(std::forward<A>(arg)), katana);
    }
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_slice slice;

} // end anonymous namespace

} // end ubu

#include "../../../detail/epilogue.hpp"

