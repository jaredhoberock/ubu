#pragma once

#include "../../detail/prologue.hpp"

#include "executor.hpp"
#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class T>
concept has_associated_executor_member_function = requires(T arg)
{
  {arg.associated_executor()} -> executor;
};

template<class T>
concept has_associated_executor_free_function = requires(T arg)
{
  {associated_executor(arg)} -> executor;
};


// this is the type of associated_executor
struct dispatch_associated_executor
{
  // this dispatch path calls the member function
  template<class T>
    requires has_associated_executor_member_function<T&&>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).associated_executor();
  }

  // this dispatch path calls the free function
  template<class T>
    requires (!has_associated_executor_member_function<T&&> and has_associated_executor_free_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return associated_executor(std::forward<T>(arg));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_associated_executor associated_executor;

} // end anonymous namespace


template<class T>
using associated_executor_result_t = decltype(UBU_NAMESPACE::associated_executor(std::declval<T>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

