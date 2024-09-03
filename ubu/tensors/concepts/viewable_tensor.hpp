#pragma once

#include "../../detail/prologue.hpp"
#include "../views/all.hpp"
#include <utility>

namespace ubu
{

template<class T>
concept viewable_tensor =
  tensor<T>
  and requires(T t)
  {
    all(std::forward<T>(t));
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

