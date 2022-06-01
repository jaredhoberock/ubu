#pragma once

#include "../../../detail/prologue.hpp"


namespace ubu::detail
{


struct invocable_archetype
{
  template<class... Types>
  void operator()(Types&&...) const {}
};


} // end ubu::detail


#include "../../../detail/epilogue.hpp"

