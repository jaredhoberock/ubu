#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <string_view>
#include <type_traits>

namespace ubu
{

namespace detail
{

template<class T>
concept has_thread_scope_static_member_variable = requires
{
  { T::thread_scope } -> std::convertible_to<std::string_view>;
};

template<class T>
concept has_thread_scope_static_member_function = requires
{
  { T::thread_scope() } -> std::convertible_to<std::string_view>;
};

template<class T>
concept has_thread_scope_member_function = requires(T arg)
{
  { arg.thread_scope() } -> std::convertible_to<std::string_view>;
};

template<class T>
concept has_thread_scope_free_function = requires(T arg)
{
  { thread_scope(arg) } -> std::convertible_to<std::string_view>;
};


struct dispatch_thread_scope
{
  // static cases do not take a parameter
  template<class T>
    requires has_thread_scope_static_member_variable<std::remove_cvref_t<T>>
  constexpr std::string_view operator()() const
  {
    return std::remove_cvref_t<T>::thread_scope;
  }

  template<class T>
    requires (not has_thread_scope_static_member_variable<std::remove_cvref_t<T>>
              and has_thread_scope_static_member_function<std::remove_cvref_t<T>>)
  constexpr std::string_view operator()() const
  {
    return std::remove_cvref_t<T>::thread_scope();
  }

  // dynamic cases do take a parameter
  template<class T>
    requires has_thread_scope_member_function<T&&>
  constexpr std::string_view operator()(T&& arg) const
  {
    return std::forward<T>(arg).thread_scope();
  }

  template<class T>
    requires (not has_thread_scope_member_function<T&&>
              and has_thread_scope_free_function<T&&>)
  constexpr std::string_view operator()(T&& arg) const
  {
    return thread_scope(std::forward<T>(arg));
  }

  template<class T>
    requires (not has_thread_scope_member_function<T&&>
              and not has_thread_scope_free_function<T&&>
              and has_thread_scope_static_member_variable<std::remove_cvref_t<T&&>>)
  constexpr std::string_view operator()(T&&) const
  {
    return std::remove_cvref_t<T>::thread_scope;
  }
};

} // end detail


inline constexpr detail::dispatch_thread_scope thread_scope;


namespace detail
{

template<class T>
concept has_static_thread_scope = requires
{
  thread_scope.operator()<T>();
};

} // end detail


template<detail::has_static_thread_scope B>
inline constexpr const std::string_view thread_scope_v = thread_scope.operator()<B>();


} // end ubu

#include "../detail/epilogue.hpp"

