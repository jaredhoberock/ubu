#pragma once

#include "../detail/prologue.hpp"

#include "property.hpp"
#include <string>
#include <type_traits>
#include <utility>


namespace ubu
{

namespace detail
{


#if defined(__circle_lang__)
template<const char name[], class T, class P>
concept has_explicit_mix_member_function = requires(T t, P prop)
{
  { t.@("mix_" + name)(prop.value) } -> has_property<std::remove_cvref_t<P>>;
};
#endif


template<class T, class P>
concept has_mix_member_function = requires(T t, P prop)
{
  { t.mix(prop) } -> has_property<std::remove_cvref_t<P>>;
};

template<class T, class P>
concept has_mix_free_function = requires(T t, P prop)
{
  { mix(t,prop) } -> has_property<std::remove_cvref_t<P>>;
};


// this is the type of mix
struct dispatch_mix
{

#if defined(__circle_lang__)
  template<class T, class P>
    requires has_explicit_mix_member_function<std::remove_cvref_t<P>::name, T, P>
  constexpr auto operator()(T&& t, P&& prop) const
  {
    return std::forward<T>(t).@(std::string("mix_") + std::remove_cvref_t<P>::name)(std::forward<P>(prop).value);
  }
#endif

  // this dispatch path calls the member function
  template<class T, class P>
    requires (has_mix_member_function<T&&,P&&>
#if defined(__circle_lang__)
              and !has_explicit_mix_member_function<std::remove_cvref_t<P>::name, T, P>
#endif
             )
  constexpr auto operator()(T&& t, P&& prop) const
  {
    return std::forward<T>(t).mix(std::forward<P>(prop));
  }

  // this dispatch path calls the free function
  template<class T, class P>
    requires (!has_mix_member_function<T&&,P&&>
              and has_mix_free_function<T&&,P&&>
#if defined(__circle_lang__)
              and !has_explicit_mix_member_function<std::remove_cvref_t<P>::name, T, P>
#endif
             )
  constexpr auto operator()(T&& t, P&& prop) const
  {
    return mix(std::forward<T>(t), std::forward<P>(prop));
  }
};


} // end detail


constexpr detail::dispatch_mix mix{};


template<class T, class P>
concept can_mix_property = requires(T t, P prop)
{
  mix(t, prop);
};


} // end ubu

#include "../detail/epilogue.hpp"

