#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/elemental_invocable.hpp"
#include "../concepts/tensor.hpp"
#include "../concepts/view.hpp"
#include "compose.hpp"
#include "zip.hpp"


// ubu::transform is variadic and the equivalent of std::views::zip_transform
// ubu::compose can already do what std::views::transform does

namespace ubu
{
namespace detail
{

template<class A1, class A2, class... As>
concept has_transform_member_function = requires(A1 arg1, A2 arg2, As... args)
{
  { std::forward<A1>(arg1).transform(std::forward<A2>(arg2), std::forward<As>(args)...) } -> view;
};

template<class A1, class A2, class... As>
concept has_transform_free_function = requires(A1 arg1, A2 arg2, As... args)
{
  { transform(std::forward<A1>(arg1), std::forward<As>(args)...) } -> view;
};

template<class A1, class A2, class... As>
concept has_transform_customization = has_transform_member_function<A1,A2,As...> or has_transform_free_function<A1,A2,As...>;


struct dispatch_transform
{
  template<class A1, class A2, class... As>
    requires has_transform_customization<A1&&,A2&&,As&&...>
  constexpr view auto operator()(A1&& arg1, A2&& arg2, As&&... args) const
  {
    if constexpr (has_transform_member_function<A1&&,A2&&,As&&...>)
    {
      return std::forward<A1>(arg1).transform(std::forward<A2&&>(arg2), std::forward<As>(args)...);
    }
    else
    {
      return transform(std::forward<A1>(arg1), std::forward<A2&&>(arg2), std::forward<As>(args)...);
    }
  }

  template<tensor T, elemental_invocable<T&&> F>
    requires (not has_transform_customization<T&&,F>)
  constexpr view auto operator()(T&& t, F f) const
  {
    return compose(f, std::forward<T>(t));
  }

  template<tensor T1, tensor T2, elemental_invocable<T1&&,T2&&> F>
    requires (not has_transform_customization<T1&&,T2&&,F>)
  constexpr view auto operator()(T1&& t1, T2&& t2, F f) const
  {
    auto g = [f](auto tuple)
    {
      return tuples::unpack_and_invoke(tuple, f);
    };

    auto zipped = zip(std::forward<T1>(t1), std::forward<T2>(t2));

    return compose(g, zipped);
  }

  template<tensor T1, tensor T2, tensor T3, elemental_invocable<T1&&,T2&&,T3&&> F>
    requires (not has_transform_customization<T1&&,T2&&,T3&&,F>)
  constexpr view auto operator()(T1&& t1, T2&& t2, T3&& t3, F f) const
  {
    auto g = [f](auto tuple)
    {
      return tuples::unpack_and_invoke(tuple, f);
    };

    auto zipped = zip(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3));

    return compose(g, zipped);
  }

  template<tensor T1, tensor T2, tensor T3, tensor T4, elemental_invocable<T1&&,T2&&,T3&&,T4&&> F>
    requires (not has_transform_customization<T1&&,T2&&,T3&&,T4&&,F>)
  constexpr view auto operator()(T1&& t1, T2&& t2, T3&& t3, T4&& t4, F f) const
  {
    auto g = [f](auto tuple)
    {
      return tuples::unpack_and_invoke(tuple, f);
    };

    auto zipped = zip(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3), std::forward<T4>(t4));

    return compose(g, zipped);
  }
};


} // end detail

inline constexpr detail::dispatch_transform transform;

} // end ubu

#include "../../detail/epilogue.hpp"

