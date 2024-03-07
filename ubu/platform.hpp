#pragma once

#include "detail/prologue.hpp"

#include "platform/cpp.hpp"
#if __has_include("cuda_runtime_api.h")
#include "platform/cuda.hpp"
#endif

#include "detail/epilogue.hpp"


