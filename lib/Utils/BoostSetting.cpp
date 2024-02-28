#include "toucan/ToucanUtils.h"
#include "llvm/Support/Debug.h"


void boost::throw_exception(std::exception const & e){
  llvm::dbgs() << e.what();
  assert(false);
}

void boost::throw_exception(std::exception const & e, boost::source_location const &loc){
  llvm::dbgs() << loc.to_string() << e.what();
  assert(false);
}