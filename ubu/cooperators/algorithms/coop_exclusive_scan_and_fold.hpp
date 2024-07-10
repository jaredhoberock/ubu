#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/cooperator.hpp"
#include "../primitives/broadcast.hpp"
#include "../primitives/last_id.hpp"
#include "coop_exclusive_scan.hpp"
#include <concepts>
#include <optional>
#include <utility>

namespace ubu
{
namespace detail
{

template<class S, class I, class T, class F>
concept has_coop_exclusive_scan_and_fold_with_init_member_function = requires(S self, I init, T value, F op)
{
  { self.coop_exclusive_scan_and_fold(init, value, op) } -> tuples::pair_like;
};

template<class S, class I, class T, class F>
concept has_coop_exclusive_scan_and_fold_with_init_free_function = requires(S self, I init, T value, F op)
{
  { coop_exclusive_scan_and_fold(self, init, value, op) } -> tuples::pair_like;
};

template<class S, class T, class F>
concept has_coop_exclusive_scan_and_fold_without_init_member_function = requires(S self, T value, F op)
{
  { self.coop_exclusive_scan_and_fold(value, op) } -> tuples::pair_like;
};

template<class S, class T, class F>
concept has_coop_exclusive_scan_and_fold_without_init_free_function = requires(S self, T value, F op)
{
  { coop_exclusive_scan_and_fold(self, value, op) } -> tuples::pair_like;
};

struct dispatch_coop_exclusive_scan_and_fold
{
  // these are with-init paths
  template<class S, class I, class T, class F>
    requires has_coop_exclusive_scan_and_fold_with_init_member_function<S&&,I&&,T&&,F&&>
  constexpr tuples::pair_like auto operator()(S&& self, I&& init, T&& value, F&& op) const
  {
    return std::forward<S>(self).coop_exclusive_scan_and_fold(std::forward<I>(init), std::forward<T>(value), std::forward<F>(op));
  }

  template<class S, class I, class T, class F>
    requires (not has_coop_exclusive_scan_and_fold_with_init_member_function<S&&,I&&,T&&,F&&> and
                  has_coop_exclusive_scan_and_fold_with_init_free_function<S&&,I&&,T&&,F&&>)
  constexpr tuples::pair_like auto operator()(S&& self, I&& init, T&& value, F&& op) const
  {
    return coop_exclusive_scan_and_fold(std::forward<S>(self), std::forward<I>(value), std::forward<T>(value), std::forward<F>(op));
  }

  template<cooperator S, class T, std::invocable<T,T> F>
    requires (not has_coop_exclusive_scan_and_fold_with_init_member_function<S&&,std::optional<T>,std::optional<T>,F> and
              not has_coop_exclusive_scan_and_fold_with_init_free_function<S&&,std::optional<T>,std::optional<T>,F>)
  constexpr tuples::pair_like auto operator()(S&& self, std::optional<T> init, std::optional<T> value, F op) const
  {
    // the default implementation is simply coop_exclusive_scan + broadcast

    std::optional exclusive_scan_result = coop_exclusive_scan(self, init, value, op);

    std::optional fold = exclusive_scan_result;
    if(fold and value)
    {
      fold = op(*fold, *value);
    }
    else if(value)
    {
      fold = value;
    }

    fold = broadcast(self, last_id(self), fold);

    return std::pair(exclusive_scan_result, fold);
  }


  // these are without-init paths
  template<class S, class T, class F>
    requires has_coop_exclusive_scan_and_fold_without_init_member_function<S&&,T&&,F&&>
  constexpr tuples::pair_like auto operator()(S&& self, T&& value, F&& op) const
  {
    return std::forward<S>(self).coop_exclusive_scan_and_fold(std::forward<T>(value), std::forward<F>(op));
  }

  template<class S, class T, class F>
    requires (not has_coop_exclusive_scan_and_fold_without_init_member_function<S&&,T&&,F&&> and
                  has_coop_exclusive_scan_and_fold_without_init_free_function<S&&,T&&,F&&>)
  constexpr tuples::pair_like auto operator()(S&& self, T&& value, F&& op) const
  {
    return coop_exclusive_scan_and_fold(std::forward<S>(self), std::forward<T>(value), std::forward<F>(op));
  }

  template<cooperator S, class T, std::invocable<T,T> F>
    requires (not has_coop_exclusive_scan_and_fold_without_init_member_function<S&&,std::optional<T>,F> and
              not has_coop_exclusive_scan_and_fold_without_init_free_function<S&&,std::optional<T>,F>)
  constexpr tuples::pair_like auto operator()(S&& self, std::optional<T> value, F op) const
  {
    // the default implementation is simply coop_exclusive_scan + broadcast

    std::optional exclusive_scan_result = coop_exclusive_scan(self, value, op);

    std::optional fold = exclusive_scan_result;
    if(fold and value)
    {
      fold = op(*fold, *value);
    }
    else if(value)
    {
      fold = value;
    }

    fold = broadcast(self, last_id(self), fold);

    return std::pair(exclusive_scan_result, fold);
  }
};

} // end detail

constexpr inline detail::dispatch_coop_exclusive_scan_and_fold coop_exclusive_scan_and_fold;

} // end ubu

#include "../../detail/epilogue.hpp"

