
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <filesystem>
#include <fstream>
#include <format>
#include <vector>


#define GEN_PASS_DEF_CPUSINGLETHREADCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/CodeGenCommon.h"
#include "ToucanGenDataTypes.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "CPUSingleThreadCodeGenPass"

struct CPUSingleThreadCodeGenPass : toucan::impl::CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>, CodeGenHelper {
  using CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>::CPUSingleThreadCodeGenBase;


  size_t indentSize = 2;
  // std::string className = "SimDesign";

  static std::string getOpVectorName(size_t partId, size_t levelId) {
    return std::format("p{}_{}_ops", partId, levelId);
  }

  std::vector<std::string> _indentStrings;
  const std::string getIndent(size_t indentLevel) {
    if (indentLevel >= _indentStrings.size()) {
      for (size_t i = _indentStrings.size(); i <= indentLevel; i++) {
        std::string indentString(i * indentSize, ' ');
        _indentStrings.push_back(indentString);
      }
      assert(_indentStrings.size() == indentLevel + 1);
    }
    return _indentStrings[indentLevel];
  }



  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto outputFullFileName = std::filesystem::path(outputDirectory.getValue()) / outputFilename.getValue();

    std::ofstream ofs(outputFullFileName);

    auto partitionResult = getAnalysis<NaivePartitioner>();
    populateLUT();

    // toucanSim::SimDesignInfo outputInfo;





    // Write register pool
    ofs << getIndent(1) << "// todo: consider randomize registers\n";
    ofs << getIndent(1) << "std::array<uint8_t, " << partitionResult.codeGenInfo.regPool.size() << "> regPool;\n\n";

    // Write memory pool
    ofs << getIndent(1) << "// todo: consider randomize memories\n";
    ofs << getIndent(1) << "std::array<uint8_t, " << partitionResult.codeGenInfo.totalMemSize << "> memPool;\n\n";

    ofs << getIndent(1) << "bool shouldStop = false;\n\n";

    
    for (size_t partId = 0; partId < partitionResult.codeGenInfo.partitionInfo.size(); partId++) {
      auto &part = partitionResult.codeGenInfo.partitionInfo[partId];

      ofs << getIndent(1) << "std::array<uint8_t, " << part.valuePool.size() << "> p" << partId << "_valuePool;\n";
      ofs << getIndent(1) << "std::array<uint8_t, " << part.numConstsInValuePool << "> p" << partId << "_constValues = {";
      for (size_t i = 0; i < part.numConstsInValuePool; i++) {
        assert(part.valuePool[i].isConst);
        ofs << static_cast<uint32_t>(part.valuePool[i].value);
        // ofs << "0x" << toHex(part.valuePool[i].value);
        if (i != part.numConstsInValuePool - 1) {
          ofs << ", ";
        }
      }
      ofs << "};\n";

      // ops
      // first level

      ofs << getIndent(1) << "static constexpr toucanSim::CGRegReadMetaInfo ";
      ofs << getOpVectorName(partId, 0) << "[] = {\n";
      
      for (auto opMeta: part.opPool[0]) {
        assert(opMeta.opName == CGToucanOPName::RegRead);
        ofs << "{" << opMeta.regRead.reg << ", " << opMeta.regRead.result << "},\n";
      }
      ofs << getIndent(1) << "};\n\n";

      // ops, middle levels
      assert(part.opPool.size() >= 3);
      for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
        ofs << getIndent(1) << "static constexpr toucanSim::CGExecLevelMetaInfo ";
        ofs << getOpVectorName(partId, levelId) << "[] = {\n";
        for (auto opMeta: part.opPool[levelId]) {
          auto tOpName = static_cast<uint8_t>(opMeta.opName);

          ofs << "{" << static_cast<uint32_t>(tOpName);
          switch (opMeta.opName) {
          case CGToucanOPName::LUT: {
            ofs << ", {.lut = {" 
              << static_cast<uint32_t>(opMeta.lut.lutId) << ", "
              << opMeta.lut.op0 << ", "
              << opMeta.lut.op1 << ", "
              << opMeta.lut.op2 << ", "
              << opMeta.lut.result 
            << "}}";
            break;
          }
          case CGToucanOPName::VecRead: {
            ofs << ", {.vec = {" 
              << opMeta.vec.vecBase << ", "
              << opMeta.vec.vecLength << ", "
              << opMeta.vec.index0 << ", "
              << opMeta.vec.index1 << ", "
              << opMeta.vec.index2 << ", "
              << opMeta.vec.index3 << ", "
              << opMeta.vec.outRangeValue << ", "
              << opMeta.vec.offset << ", "
              << opMeta.vec.result 
            << "}}";
            break;
          }
          case CGToucanOPName::MemRead: {
            ofs << ", {.mem = {" 
              << opMeta.memRead.hasMultipleWriter << ", "
              << opMeta.memRead.memDepth << ", "
              << opMeta.memRead.memBase << ", "
              << opMeta.memRead.addr0 << ", "
              << opMeta.memRead.addr1 << ", "
              << opMeta.memRead.addr2 << ", "
              << opMeta.memRead.addr3 << ", "
              << opMeta.memRead.addr4 << ", "
              << opMeta.memRead.addr5 << ", "
              << opMeta.memRead.addr6 << ", "
              << opMeta.memRead.addr7 << ", "
              << opMeta.memRead.result
            << "}}";
            break;
          }
          default:
            llvm::dbgs() << static_cast<uint32_t>(opMeta.opName);
            llvm_unreachable("Other ops should not appear here");
          }
          ofs << "},\n";
        }
        ofs << getIndent(1) << "};\n\n";
      }

      // ops, last level
      // ofs << getIndent(1) << "std::vector<toucanSim::CGLastLevelMetaInfo> ";
      ofs << getIndent(1) << "static constexpr toucanSim::CGLastLevelMetaInfo ";
      ofs << getOpVectorName(partId, part.opPool.size() - 1) << "[] = {\n";
      for (auto opMeta: part.opPool.back()) {
        auto tOpName = static_cast<uint8_t>(opMeta.opName);
        ofs << "{" << static_cast<uint32_t>(tOpName);
        switch (opMeta.opName) {

        case CGToucanOPName::Print: {
          ofs << ", {.print = {"
            << opMeta.print.en << ", "
            << opMeta.print.msg
          << "}}";
          break;
        }
        case CGToucanOPName::Stop: {
          ofs << ", {.stop = {"
            << opMeta.stop.en
          << "}}";
          break;
        }
        case CGToucanOPName::RegWrite: {
          ofs << ", {.regWrite = {"
            << opMeta.regWrite.reg << ", "
            << opMeta.regWrite.dat
          << "}}";
          break;
        }
        case CGToucanOPName::MemWrite: {
          ofs << ", {.memWrite = {" 
            << opMeta.memWrite.hasMultipleWriter << ", "
            << opMeta.memWrite.memDepth << ", "
            << opMeta.memWrite.memBase << ", "
            << opMeta.memWrite.addr0 << ", "
            << opMeta.memWrite.addr1 << ", "
            << opMeta.memWrite.addr2 << ", "
            << opMeta.memWrite.addr3 << ", "
            << opMeta.memWrite.addr4 << ", "
            << opMeta.memWrite.addr5 << ", "
            << opMeta.memWrite.addr6 << ", "
            << opMeta.memWrite.addr7 << ", "
            << opMeta.memWrite.dat << ", "
            << opMeta.memWrite.en
          << "}}";
          break;
        }
        default:
          llvm_unreachable("Other type of ops should not appear in last level!");
        }
        ofs << "},\n";
      }
      ofs << getIndent(1) << "};\n\n";


      // pointers to all levels
      ofs << getIndent(1) << "static constexpr toucanSim::CGExecLevelMetaInfo* p0_middleLevels[] = {\n";
      for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
        ofs << getIndent(1) << getOpVectorName(partId, levelId) << ", \n";
      }
      ofs << getIndent(1) << "};\n";

      // level size
      ofs << getIndent(1) << "static constexpr uint32_t p0_middleLevelSize[] = {\n";
      for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
        ofs << part.opPool[levelId].size() << ", ";
      }
      ofs << getIndent(1) << "};\n";
      
    }


    // Write top level
    ofs << "// First level operations. Only register reads\n";

    // 1. write lut

    // 2. declare data

    // 3. write netlist

    // 4. write signal name


    // 5. write eval
    // write init func


    ofs << "};\n\n";

    ofs.close();
    

  }

};

std::unique_ptr<mlir::Pass> toucan::createCPUSingleThreadCodeGenPass(CPUSingleThreadCodeGenOptions option) {
  return std::make_unique<CPUSingleThreadCodeGenPass>(option);
}