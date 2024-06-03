#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../tensor/shape/convert_shape.hpp"
#include "../../tensor/views/compose.hpp"
#include "../../tensor/views/layout/identity_layout.hpp"
#include "../../tensor/views/layout/lifting_layout.hpp"
#include "../../tensor/views/layout/native_layout.hpp"
#include "concepts/executor.hpp"
#include "traits/executor_coordinate.hpp"


namespace ubu
{

namespace detail
{

template<executor E, std::invocable<executor_coordinate_t<E>> F>
constexpr layout auto default_kernel_layout(const E&, const executor_coordinate_t<E>& shape, F&&)
{
  // when the type of shape requested by the user is the same type as the coordinates produced
  // by the executor, we just return an identity_layout
  return identity_layout(shape);
}

// postcondition: shape_size(result.coshape()) >= shape_size(shape)
// postcondition: layout_coshape_t<decltype(result)> == S
// XXX it's probably possible to constrain the result layout a bit more
template<executor E, coordinate S, std::invocable<S> F>
  requires (not std::same_as<S,executor_coordinate_t<E>>)
constexpr layout auto default_kernel_layout(const E& ex, const S& shape, F&&)
{
  // convert the user's requested shape into the native shape of the executor
  using native_coord_t = executor_coordinate_t<E>;
  native_coord_t native_shape = convert_shape<native_coord_t>(shape);

  // create a layout that maps a native coordinate to an integer
  // XXX we need to require that the result of native_layout has a .coshape() that is integral
  layout auto native_coord_to_index = native_layout(ex, native_shape);

  // create a layout that maps an integer to a coordinate in shape
  lifting_layout index_to_user_coord(native_coord_to_index.coshape(), shape);

  // the kernel layout is the composition of these two layouts
  return compose(index_to_user_coord, native_coord_to_index);
}


template<class E, class S, class F>
concept has_kernel_layout_member_function = requires(E ex, S shape, F f)
{
  { ex.kernel_layout(shape, f) } -> layout;
};

template<class E, class S, class F>
concept has_kernel_layout_free_function = requires(E ex, S shape, F f)
{
  { kernel_layout(ex, shape, f) } -> layout;
};


// this is the type of kernel_layout
// postcondition: shape_size(result.coshape()) >= shape_size(shape)
// postcondition: layout_coshape_t<decltype(result)> == S
struct dispatch_kernel_layout
{
  // this dispatch path calls the member function
  template<class E, class S, class F>
    requires has_kernel_layout_member_function<E&&,S&&,F&&>
  constexpr layout auto operator()(E&& ex, S&& shape, F&& f) const
  {
    return std::forward<E>(ex).kernel_layout(std::forward<S>(shape), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class S, class F>
    requires (not has_kernel_layout_member_function<E&&,S&&,F&&>
              and has_kernel_layout_free_function<E&&,S&&,F&&>)
  constexpr layout auto operator()(E&& ex, S&& shape, F&& f) const
  {
    return kernel_layout(std::forward<E>(ex), std::forward<S>(shape), std::forward<F>(f));
  }

  // this dispatch path calls the default implementation
  template<executor E, coordinate S, std::invocable<S> F>
    requires (not has_kernel_layout_member_function<const E&,const S&,F&&>
              and not has_kernel_layout_free_function<const E&,const S&,F&&>)
  constexpr layout auto operator()(const E& ex, const S& shape, F&& f) const
  {
    return default_kernel_layout(ex, shape, std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_kernel_layout kernel_layout;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

