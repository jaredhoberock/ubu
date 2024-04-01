#pragma once

#include "../../detail/prologue.hpp"
#include "synchronize_and_count.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{

template<class C, class B>
concept has_synchronize_and_any_member_function = requires(C self, B value)
{
  { self.synchronize_and_any(static_cast<bool>(value)) } -> std::convertible_to<bool>;
};

template<class C, class B>
concept has_synchronize_and_any_free_function = requires(C self, B value)
{
  { synchronize_and_any(self, static_cast<bool>(value)) } -> std::convertible_to<bool>;
};

template<class C, class B>
concept has_synchronize_and_any_customization = has_synchronize_and_any_member_function<C,B> and has_synchronize_and_any_free_function<C,B>;

// this is the type of synchronize_and_any
struct dispatch_synchronize_and_any
{
  template<class C, class B>
    requires has_synchronize_and_any_customization<C&&,B&&>
  constexpr std::convertible_to<bool> auto operator()(C&& self, B&& value) const
  {
    if constexpr(has_synchronize_and_any_member_function<C&&,B&&>)
    {
      return std::forward<C>(self).synchronize_and_any(std::forward<B>(value));
    }
    else if constexpr(has_synchronize_and_any_free_function<C&&,B&&>)
    {
      return synchronize_and_any(std::forward<C>(self), std::forward<B>(value));
    }
    else
    {
      // cast value to int and count the number of non-zero values
      return synchronize_and_count(std::forward<C>(self), static_cast<int>(std::forward<B>(value))) > 0;
    }
  }
};

} // end detail

inline constexpr detail::dispatch_synchronize_and_any synchronize_and_any;

} // end ubu

#include "../../detail/epilogue.hpp"

