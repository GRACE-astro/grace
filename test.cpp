#include <array>
#include <vector>

int main() {

  using arr_t = std::array<std::array<int,2>,3> ;
  using vec_t = std::vector<arr_t> ;

  vec_t A ;

  A.resize(10, {{{{0,0}},{{0,0}},{{0,0}}}}) ; 
  
}
