#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/concepts/congruent.hpp"
#include "../concepts/nested_tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../traits/inner_tensor_shape.hpp"
#include "../quilted_tensor.hpp"
#include <utility>


namespace ubu
{

// the only difference between quilted_view and quilted_tensor is that quilted_view
// requires P to be a view
template<nested_tensor_like P, congruent<inner_tensor_shape_t<P>> S = inner_tensor_shape_t<P>>
  requires view<P>
class quilted_view : public quilted_tensor<P,S>
{
  private:
    using super_t = quilted_tensor<P,S>;

  public:
    using super_t::super_t;
};


namespace detail
{


template<class T, class S>
concept has_quilt_member_function = requires(T t, S s)
{
  { std::forward<T>(t).quilt(std::forward<S>(s)) } -> view;
};

template<class T, class S>
concept has_quilt_free_function = requires(T t, S s)
{
  { quilt(std::forward<T>(t), std::forward<S>(s)) } -> view;
};

template<class T, class S>
concept has_quilt_customization = 
  has_quilt_member_function<T,S>
  or has_quilt_free_function<T,S>
;


struct dispatch_quilt
{
  template<class T, class S>
    requires has_quilt_customization<T&&,S&&>
  constexpr view auto operator()(T&& t, S&& s) const
  {
    if constexpr (has_quilt_member_function<T&&,S&&>)
    {
      return std::forward<T>(t).quilt(std::forward<S>(s));
    }
    else
    {
      return quilt(std::forward<T>(t), std::forward<S>(s));
    }
  }

  template<nested_tensor_like T, congruent<inner_tensor_shape_t<T&&>> S>
    requires (not has_quilt_customization<T&&,S>) const
  constexpr view auto operator()(T&& t, S s) const
  {
    return quilted_view(all(std::forward<T>(t)), s);
  }
};


} // end detail


inline constexpr detail::dispatch_quilt quilt;


template<class T, class S>
using quilt_t = decltype(quilt(std::declval<T>(), std::declval<S>()));


} // end ubu

#include "../../detail/epilogue.hpp"
