#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <future>
#include <utility>


namespace ubu
{

namespace detail
{


template<class H>
concept has_has_happened_member_function = requires(H h)
{
  { h.has_happened() } -> std::convertible_to<bool>;
};

template<class H>
concept has_has_happened_free_function = requires(H h)
{
  { has_happened(h) } -> std::convertible_to<bool>;
};


// this is the type of has_happened
struct dispatch_has_happened
{
  // this dispatch path calls the member function
  template<class H>
    requires has_has_happened_member_function<H&&>
  constexpr auto operator()(H&& h) const
  {
    return std::forward<H>(h).has_happened();
  }

  // this dispatch path calls the free function
  template<class H>
    requires (!has_has_happened_member_function<H&&> and has_has_happened_free_function<H&&>)
  constexpr auto operator()(H&& h) const
  {
    return has_happened(std::forward<H>(h));
  }


  // customization for std::future<void>
  inline bool operator()(const std::future<void>& f) const
  {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_has_happened has_happened;

} // end anonymous namespace


template<class H>
using has_happened_result_t = decltype(ubu::has_happened(std::declval<H>()));


} // end ubu


#include "../detail/epilogue.hpp"


