#pragma once

#include "../../detail/prologue.hpp"

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


struct invocable_archetype
{
  template<class... Types>
  void operator()(Types&&...) const {}
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

