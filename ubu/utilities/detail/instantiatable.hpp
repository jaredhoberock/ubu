#pragma once

#include "../../detail/prologue.hpp"

namespace ubu::detail
{

// checks whether some template can be instantiated with some list of types
template<template<class...> class Template, class... Types>
concept instantiatable = requires
{
  typename Template<Types...>;
};


} // end ubu::detail

#include "../../detail/epilogue.hpp"

