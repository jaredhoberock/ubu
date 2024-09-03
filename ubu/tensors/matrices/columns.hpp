#pragma once

#include "../../detail/prologue.hpp"
#include "../vectors/vector_like.hpp"
#include "../views/nestle.hpp"
#include "matrix.hpp"
#include <utility>

namespace ubu
{

template<matrix M>
constexpr vector_like auto columns(M&& m)
{
  return nestle(std::forward<M>(m));
}

} // end ubu

#include "../../detail/epilogue.hpp"

