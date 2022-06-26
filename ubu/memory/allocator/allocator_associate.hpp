#pragma once

#include "../../detail/prologue.hpp"

#include "associated_allocator.hpp"


namespace ubu
{


template<class T>
concept allocator_associate = requires(T arg)
{
  ubu::associated_allocator(arg);
};


} // end ubu

#include "../../detail/epilogue.hpp"

