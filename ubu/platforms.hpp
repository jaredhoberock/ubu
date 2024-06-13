#pragma once

#include "detail/prologue.hpp"

#include "platforms/cpp.hpp"
#if __has_include("cuda_runtime_api.h")
#include "platforms/cuda.hpp"
#endif

#include "detail/epilogue.hpp"


