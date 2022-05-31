#pragma once

#include "../../../detail/prologue.hpp"

UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


struct invocable_archetype
{
  template<class... Types>
  void operator()(Types&&...) const {}
};


} // end detail


UBU_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

