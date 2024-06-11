#include <ubu/ubu.hpp>

#ifdef UBU_INCLUDE_LEVEL
#error UBU_INCLUDE_LEVEL defined in client code.
#endif

#ifdef UBU_TARGET
#error UBU_TARGET defined in client code.
#endif

#ifdef UBU_TARGET_NEEDS_UNDEF
#error UBU_TARGET_NEEDS_UNDEF defined in client code.
#endif

#ifdef UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF
#error UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF defined in client code.
#endif

void test_preprocessor() {}

