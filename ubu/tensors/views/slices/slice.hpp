#pragma once

#include "../../../detail/prologue.hpp"
#include "../../concepts/decomposable.hpp"
#include "../../concepts/tensor.hpp"
#include "../../concepts/view.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "../../traits/tensor_reference.hpp"
#include "../../traits/tensor_shape.hpp"
#include "../all.hpp"
#include "../compose.hpp"
#include "slice_coordinate.hpp"
#include "slicing_layout.hpp"
#include "sliceable_with.hpp"
#include "slicer.hpp"
#include "underscore.hpp"

namespace ubu
{
namespace detail
{


template<class R, class T, class K>
concept view_of_slice =
  view<R>
  and sliceable_with<T,K>
  and congruent<tensor_shape_t<R>, slice_coordinate_result_t<tensor_shape_t<T>,K>>
  and std::same_as<tensor_reference_t<R>, tensor_reference_t<T>>
;


template<class T, class K>
concept has_slice_member_function = requires(T arg, K katana)
{
  { arg.slice(katana) } -> view_of_slice<T,K>;
};

template<class T, class K>
concept has_slice_free_function = requires(T arg, K katana)
{
  { slice(arg,katana) } -> view_of_slice<T,K>;
};


template<class T, class K>
concept has_slice_customization = has_slice_member_function<T,K> or has_slice_free_function<T,K>;


// slice and slicing_layout have a cyclic dependency and can't use each other directly
// declare detail::make_slicing_layout for slice's use
template<coordinate S, slicer K, coordinate R>
  requires congruent<R,unslice_coordinate_result_t<S,K>>
constexpr view auto make_slicing_layout(S shape, K katana, R coshape);


struct dispatch_slice
{
  template<class A, class K>
    requires has_slice_customization<A&&,K&&>
  constexpr view_of_slice<A,K> auto operator()(A&& arg, K&& katana) const
  {
    if constexpr (has_slice_member_function<A&&,K&&>)
    {
      return std::forward<A>(arg).slice(std::forward<K>(katana));
    }
    else
    {
      return slice(std::forward<A>(arg), std::forward<K>(katana));
    }
  }

  template<class K, sliceable_with<K> A>
    requires (not has_slice_customization<A&&,K>)
  constexpr view_of_slice<A,K> auto operator()(A&& arg, K katana) const
  {
    if constexpr (decomposable<A&&>)
    {
      // decompose a into left & right views
      auto [left, right] = decompose(std::forward<A>(arg));

      // get a nice name for this CPO
      auto slice = *this;

      // recursively slice A's right part and compose that result with A's left part
      return compose(left, slice(right, katana));
    }
    else
    {
      auto result_coshape = shape(arg);
      auto result_shape = slice_coordinate(result_coshape, katana);

      return compose(std::forward<A>(arg), make_slicing_layout(result_shape, katana, result_coshape));
    }
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_slice slice;

} // end anonymous namespace


namespace detail
{

// slice and slicing_layout have a cyclic dependency and can't use each other directly
// define detail::invoke_slice as soon as slice's definition is available
template<class... Args>
constexpr view auto invoke_slice(Args&&... args)
{
  return slice(std::forward<Args>(args)...);
}

} // end detail


} // end ubu

#include "../../../detail/epilogue.hpp"

