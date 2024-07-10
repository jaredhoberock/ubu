#pragma once

#include "../../detail/prologue.hpp"
#include "../primitives/concepts/cooperator.hpp"
#include <concepts>
#include <optional>
#include <utility>

namespace ubu
{
namespace detail
{

template<class S, class I, class T, class F>
concept has_coop_exclusive_scan_with_init_member_function = requires(S self, I init, T value, F op)
{
  self.coop_exclusive_scan(init, value, op);
};

template<class S, class I, class T, class F>
concept has_coop_exclusive_scan_with_init_free_function = requires(S self, I init, T value, F op)
{
  coop_exclusive_scan(self, init, value, op);
};

template<class S, class T, class F>
concept has_coop_exclusive_scan_without_init_member_function = requires(S self, T value, F op)
{
  self.coop_exclusive_scan(value, op);
};

template<class S, class T, class F>
concept has_coop_exclusive_scan_without_init_free_function = requires(S self, T value, F op)
{
  coop_exclusive_scan(self, value, op);
};

struct dispatch_coop_exclusive_scan
{
  // these are with-init paths
  template<class S, class I, class T, class F>
    requires has_coop_exclusive_scan_with_init_member_function<S&&,I&&,T&&,F&&>
  constexpr auto operator()(S&& self, I&& init, T&& value, F&& op) const
  {
    return std::forward<S>(self).coop_exclusive_scan(std::forward<I>(init), std::forward<T>(value), std::forward<F>(op));
  }

  template<class S, class I, class T, class F>
    requires (not has_coop_exclusive_scan_with_init_member_function<S&&,I&&,T&&,F&&> and
                  has_coop_exclusive_scan_with_init_free_function<S&&,I&&,T&&,F&&>)
  constexpr auto operator()(S&& self, I&& init, T&& value, F&& op) const
  {
    return coop_exclusive_scan(std::forward<S>(self), std::forward<I>(init), std::forward<T>(value), std::forward<F>(op));
  }

  template<cooperator S, class I, class T, std::invocable<T,T> F>
    requires (not has_coop_exclusive_scan_with_init_member_function<S&&,I&&,T&&,F&&> and
              not has_coop_exclusive_scan_with_init_free_function<S&&,I&&,T&&,F&&>)
  constexpr auto operator()(S&& self, const std::optional<I>& init, const std::optional<T>& value, F&& op) const
  {
    static_assert(sizeof(self) == 0, "Default coop_exclusive_scan with init unimplemented.");
    return std::optional<T>();
  }


  // these are without-init paths
  template<class S, class T, class F>
    requires has_coop_exclusive_scan_without_init_member_function<S&&,T&&,F&&>
  constexpr auto operator()(S&& self, T&& value, F&& op) const
  {
    return std::forward<S>(self).coop_exclusive_scan(std::forward<T>(value), std::forward<F>(op));
  }

  template<class S, class T, class F>
    requires (not has_coop_exclusive_scan_without_init_member_function<S&&,T&&,F&&> and
                  has_coop_exclusive_scan_without_init_free_function<S&&,T&&,F&&>)
  constexpr auto operator()(S&& self, T&& value, F&& op) const
  {
    return coop_exclusive_scan(std::forward<S>(self), std::forward<T>(value), std::forward<F>(op));
  }

  template<cooperator S, class T, std::invocable<T,T> F>
    requires (not has_coop_exclusive_scan_without_init_member_function<S&&,T&&,F&&> and
              not has_coop_exclusive_scan_without_init_free_function<S&&,T&&,F&&>)
  constexpr auto operator()(S&& self, const std::optional<T>& value, F&& op) const
  {
    // use an empty optional as the init and try the with-init paths
    return (*this)(std::forward<S>(self), std::optional<T>(), value, std::forward<F>(op));
  }
};

} // end detail

constexpr inline detail::dispatch_coop_exclusive_scan coop_exclusive_scan;

} // end ubu

#include "../../detail/epilogue.hpp"

