#pragma once

#include "../../detail/prologue.hpp"
#include "../vectors/vector_like.hpp"
#include "../views/nestle.hpp"
#include "matrix_like.hpp"
#include <utility>

namespace ubu
{

template<matrix_like M>
constexpr vector_like auto rows(M&& m)
{
  return nestle(transpose(std::forward<M>(m)));
}

} // end ubu

#include "../../detail/epilogue.hpp"
