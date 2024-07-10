#pragma once

#include "../../detail/prologue.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_synchronize_member_function = requires(T arg)
{
  arg.synchronize();
};

template<class T>
concept has_synchronize_free_function = requires(T arg)
{
  synchronize(arg);
};

struct dispatch_synchronize
{
  template<class T>
    requires has_synchronize_member_function<T&&>
  constexpr void operator()(T&& arg) const
  {
    std::forward<T>(arg).synchronize();
  }

  template<class T>
    requires (not has_synchronize_member_function<T&&>
              and has_synchronize_free_function<T&&>)
  constexpr void operator()(T&& arg) const
  {
    synchronize(std::forward<T>(arg));
  }
};

} // end detail

constexpr detail::dispatch_synchronize synchronize;

} // end ubu

#include "../../detail/epilogue.hpp"

