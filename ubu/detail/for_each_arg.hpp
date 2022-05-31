#pragma once

#include "prologue.hpp"

#include "for_each_tuple_element.hpp"
#include <tuple>
#include <utility>

UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class F, class... Args>
constexpr void for_each_arg(F&& f, Args&&... args)
{
  detail::for_each_tuple_element(std::forward<F>(f), std::forward_as_tuple(std::forward<Args>(args)...));
}


} // end detail


UBU_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

