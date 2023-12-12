#pragma once

#include "../../detail/prologue.hpp"

#include "downloader.hpp"
#include "uploader.hpp"

namespace ubu
{

template<class L>
concept loader = uploader<L> and downloader<L>;

} // end ubu

#include "../../detail/epilogue.hpp"

