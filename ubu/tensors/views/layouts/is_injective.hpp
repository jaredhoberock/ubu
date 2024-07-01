#pragma once

#include "../../../detail/prologue.hpp"

#include "../../traits/tensor_element.hpp"
#include "../domain.hpp"
#include "concepts/layout_like.hpp"
#include <set>

namespace ubu
{

// XXX this is provided for debugging purposes
// the implementation is an exhaustive search
template<layout_like L>
constexpr bool is_injective(L l)
{
  std::set<tensor_element_t<L>> codomain;

  for(auto coord : domain(l))
  {
    auto image = l[coord];

    if(codomain.contains(image)) return false;

    codomain.insert(image);
  }

  return true;
}

} // end ubu

#include "../../../detail/epilogue.hpp"

