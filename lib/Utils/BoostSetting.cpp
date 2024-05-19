#include "toucan/ToucanUtils.h"
#include "llvm/Support/Debug.h"


void boost::throw_exception(std::exception const & e){
  llvm::dbgs() << e.what();
  assert(false);
}

void boost::throw_exception(std::exception const & e, boost::source_location const &loc){
  llvm::dbgs() << "File: " << loc.file_name() << "\n";
  llvm::dbgs() << "Line: " << loc.line() << "\n";
  llvm::dbgs() << "Column: " << loc.column() << "\n";
  llvm::dbgs() << "Function: " << loc.function_name() << "\n";

  llvm::dbgs() << "Error: " << e.what() << "\n";

  assert(false);
}