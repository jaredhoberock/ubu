#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/elemental_invocable.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "domain.hpp"
#include "transform.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


template<class T, class Arg1, class... Args>
concept has_enumerate_transform_member_function = requires(T t, Arg1 arg1, Args... args)
{
  { std::forward<T>(t).enumerate_transform(std::forward<Arg1>(arg1), std::forward<Args>(args)...) } -> view;
};

template<class T, class Arg1, class... Args>
concept has_enumerate_transform_free_function = requires(T t, Arg1 arg1, Args... args)
{
  { enumerate_transform(std::forward<T>(t), std::forward<Arg1>(arg1), std::forward<Args>(args)...) } -> view;
};

template<class T, class Arg1, class... Args>
concept has_enumerate_transform_customization =
  has_enumerate_transform_member_function<T,Arg1,Args...>
  or has_enumerate_transform_free_function<T,Arg1,Args...>
;


struct dispatch_enumerate_transform
{
  template<class T, class Arg1, class... Args>
    requires has_enumerate_transform_customization<T&&,Arg1&&,Args&&...>
  constexpr view auto operator()(T&& t, Arg1&& arg1, Args&&... args) const
  {
    if constexpr (has_enumerate_transform_member_function<T&&,Arg1&&,Args&&...>)
    {
      return std::forward<T>(t).enumerate_transform(std::forward<Arg1>(arg1), std::forward<Args>(args)...);
    }
    else
    {
      return enumerate_transform(std::forward<T>(t), std::forward<Arg1>(arg1), std::forward<Args>(args)...);
    }
  }

  template<tensor_like T1, elemental_invocable<domain_t<T1&&>,T1&&> F>
    requires (not has_enumerate_transform_customization<T1&&,F&&>)
  constexpr view auto operator()(T1&& t1, F&& f) const
  {
    return transform(domain(std::forward<T1>(t1)), std::forward<T1>(t1), std::forward<F>(f));
  }

  template<tensor_like T1, tensor_like T2, elemental_invocable<domain_t<T1&&>,T1&&,T2&&> F>
    requires (not has_enumerate_transform_customization<T1&&,T2&&,F&&>)
  constexpr view auto operator()(T1&& t1, T2&& t2, F&& f) const
  {
    return transform(domain(std::forward<T1>(t1)), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<F>(f));
  }

  template<tensor_like T1, tensor_like T2, tensor_like T3, elemental_invocable<domain_t<T1&&>,T1&&,T2&&,T3&&> F>
    requires (not has_enumerate_transform_customization<T1&&,T2&&,T3&&,F&&>)
  constexpr view auto operator()(T1&& t1, T2&& t2, T3&& t3, F&& f) const
  {
    return transform(domain(std::forward<T1>(t1)), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3), std::forward<F>(f));
  }

  template<tensor_like T1, tensor_like T2, tensor_like T3, tensor_like T4, elemental_invocable<domain_t<T1&&>,T1&&,T2&&,T3&&,T4&&> F>
    requires (not has_enumerate_transform_customization<T1&&,T2&&,T3&&,T4&&,F&&>)
  constexpr view auto operator()(T1&& t1, T2&& t2, T3&& t3, T4&& t4, F&& f) const
  {
    return transform(domain(std::forward<T1>(t1)), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3), std::forward<T4>(t4), std::forward<F>(f));
  }
};


} // end detail


inline constexpr detail::dispatch_enumerate_transform enumerate_transform;


} // end ubu

#include "../../detail/epilogue.hpp"

