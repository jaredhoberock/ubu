// note that this header file is special and does not use #pragma once

// The ASPERA_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              ASPERA_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef ASPERA_REQUIRES

#  define ASPERA_CONCATENATE_IMPL(x, y) x##y

#  define ASPERA_CONCATENATE(x, y) ASPERA_CONCATENATE_IMPL(x, y)

#  define ASPERA_MAKE_UNIQUE(x) ASPERA_CONCATENATE(x, __COUNTER__)

#  define ASPERA_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define ASPERA_REQUIRES(...) ASPERA_REQUIRES_IMPL(ASPERA_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#  define ASPERA_REQUIRES_DEF_IMPL(unique_name, ...) bool unique_name, typename std::enable_if<(unique_name and __VA_ARGS__)>::type*

#  define ASPERA_REQUIRES_DEF(...) ASPERA_REQUIRES_DEF_IMPL(ASPERA_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#elif defined(ASPERA_REQUIRES)

#  ifdef ASPERA_CONCATENATE_IMPL
#    undef ASPERA_CONCATENATE_IMPL
#  endif

#  ifdef ASPERA_CONCATENATE
#    undef ASPERA_CONCATENATE
#  endif

#  ifdef ASPERA_MAKE_UNIQUE
#    undef ASPERA_MAKE_UNIQUE
#  endif

#  ifdef ASPERA_REQUIRES_IMPL
#    undef ASPERA_REQUIRES_IMPL
#  endif

#  ifdef ASPERA_REQUIRES
#    undef ASPERA_REQUIRES
#  endif

#  ifdef ASPERA_REQUIRES_DEF_IMPL
#    undef ASPERA_REQUIRES_DEF_IMPL
#  endif

#  ifdef ASPERA_REQUIRES_DEF
#    undef ASPERA_REQUIRES_DEF
#  endif

#endif

