#pragma once

#include "../detail/prologue.hpp"

#include "executor_of.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


struct invocable_archetype
{
  template<class... Types>
  void operator()(Types&&...) const {}
};


} // end detail


template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

