#pragma once

#include "../../detail/prologue.hpp"

#include "allocator.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class T>
concept has_associated_allocator_member_function = requires(T arg)
{
  {arg.associated_allocator()} -> allocator;
};

template<class T>
concept has_associated_allocator_free_function = requires(T arg)
{
  {associated_allocator(arg)} -> allocator;
};


// this is the type of associated_allocator
struct dispatch_associated_allocator
{
  // this dispatch path calls the member function
  template<class T>
    requires has_associated_allocator_member_function<T&&>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).associated_allocator();
  }

  // this dispatch path calls the free function
  template<class T>
    requires (!has_associated_allocator_member_function<T&&> and has_associated_allocator_free_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return associated_allocator(std::forward<T>(arg));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_associated_allocator associated_allocator;

} // end anonymous namespace


template<class T>
using associated_allocator_result_t = decltype(ubu::associated_allocator(std::declval<T>()));


} // end ubu


#include "../../detail/epilogue.hpp"

