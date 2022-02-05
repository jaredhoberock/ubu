#pragma once

#include "../detail/prologue.hpp"

#include "../event.hpp"
#include "future.hpp"
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

template<class T>
concept future = event<T> and requires(std::decay_t<T>& f)
{
  f.get();
};

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

