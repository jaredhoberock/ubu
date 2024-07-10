#include <ubu/utilities/print.hpp>
#include <ubu/ubu.hpp>

#include <stdexcept>
#include <string>

template<class F>
std::string collect_stdout(F f)
{
  FILE* old_stdout = stdout;
  stdout = tmpfile();

  try
  {
    f();
  }
  catch(...)
  {
    stdout = old_stdout;
    throw;
  }

  FILE* fp = stdout;
  stdout = old_stdout;

  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  
  std::string result(file_size, 'x');
  if(fread(result.data(), sizeof(char), file_size, fp) != file_size)
  {
    fclose(fp);

    throw std::runtime_error("collect_stdout: Error after fread.");
  }

  return result;
}

void print_some_output()
{
  using namespace ubu;
  
  print("hello\n"_cf);
  print("hello again: {}, {}, {}\n"_cf, 1, 2, "string");
  print("here's a float: {}\n"_cf, 1.234e-20f);
  
  ubu::int2 coord(0,1);
  print("a coordinate: {}\n"_cf, coord);
  
  print("a constant: {}\n"_cf, 13_c);
  
  ubu::strided_layout layout(ubu::int3(1,2,3));
  print("a layout: {}\n"_cf, layout);
}

void test_print()
{
  std::string result = collect_stdout(print_some_output);

  std::string expected =
    "hello\n"
    "hello again: 1, 2, string\n"
    "here's a float: 1.234e-20\n"
    "a coordinate: (0, 1)\n"
    "a constant: 13_c\n"
    "a layout: (1, 2, 3):(1_c, 1, 2)\n"
  ;

  assert(expected == result);
}

