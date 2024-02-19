// Toucan

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "toucan/ToucanDialect.h"
#include "toucan/ToucanPasses.h"

#include <filesystem>

using namespace llvm;
using namespace mlir;
using namespace circt;

enum Levels {
    CoreDialect,
    ToucanHigh,
    Toucan4B,
    ToucanFlattened
};


// commandline options


static cl::OptionCategory mainCategory("Toucan Options");
const auto LevelClasses = cl::values(
        clEnumValN(CoreDialect, "core", "CIRCT core dialect"),
        clEnumValN(ToucanHigh, "toucanHigh", "Toucan dialect, high level"),
        clEnumValN(Toucan4B, "toucan4B", "Toucan dialect, signals lowered to 4 bits"),
        clEnumValN(ToucanFlattened, "toucanFlattened", "Toucan dialect, flattened")
);


static cl::opt<Levels> inputLevel("inputLevel", cl::desc("Input file level"), LevelClasses, cl::init(CoreDialect), cl::cat(mainCategory));
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"), cl::cat(mainCategory));
static cl::opt<std::string> outputDirectory("o", cl::desc("Output directory"), cl::value_desc("directory"), cl::init("./"), cl::cat(mainCategory));
static cl::opt<bool> verbose("v", cl::desc("verbose"), cl::init(false), cl::cat(mainCategory));



static LogicalResult compileAndEmit(
        MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr, std::filesystem::path &outputDir) {

    std::string errorMsg;
    auto outputFilename = outputDir / "output.mlir";
    auto output = openOutputFile(outputFilename.string(), &errorMsg);
    if (!output) {
        llvm::errs() << errorMsg << "\n";
        return failure();
    }

    auto mod = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if(!mod) return failure();

    mlir::PassManager pm(&context);
    if(failed(applyPassManagerCLOptions(pm)))
        return failure();

    pm.enableTiming(ts);

    if (inputLevel < ToucanHigh) {
        // Lower to ToucanHigh
        
        // Expand SV macros, generates Print and Stop
        pm.addPass(toucan::createFactorSVPass());
        // Remove unsupported Ops (other SV Ops, OM Ops)
        pm.addPass(toucan::createRemoveSVnOMPass());
        // SplitRWPort
        pm.addPass(toucan::createSplitFirMemRWPortsPass());
        // Expand memory delays
        pm.addPass(toucan::createExpandMemoryDelayPass());
        // Replace async reset regs
        pm.addPass(toucan::createReplaceAsyncResetRegsPass());
        // Split registers into def, read and write nodes
        pm.addPass(toucan::createSplitRegistersPass());
        // Remove all mem write masks. Split memory if there is any mask
        pm.addPass(toucan::createRemoveMemMaskPass());

        pm.addPass(toucan::createParallelCanonicalizerPass());
    }

    if (inputLevel < Toucan4B) {
        // Lower to Toucan4B

        // Lower registers and memory to 4b
        pm.addPass(toucan::createLowerRegMemTo4BPass());

        // Remove clock and AsClock. 
        pm.addPass(toucan::createEnsureNoClockOpPass());
        // Fold binary ops with more than 2 operands
        // Convert hw array
        pm.addPass(toucan::createLowerCombPreProcessPass());

        // 2. Lower HW to 4B
        // 2.1 lowers most ops
        pm.addPass(toucan::createLowerCombTo4B_1Pass());
        pm.addPass(toucan::createLowerCombTo4B_2Pass());
        pm.addPass(toucan::createLowerCombTo4B_3Pass());

        pm.addPass(toucan::createParallelCanonicalizerPass());

        // auto option = toucan::FactorConcatExtractOptions();
        // option.parallel = true;
        // pm.addPass(toucan::createFactorConcatExtractPass(option));
    }

    if (inputLevel < ToucanFlattened) {
        // Lower to flattened
        pm.addPass(toucan::createFlattenPass());
        // pm.addPass(mlir::createCanonicalizerPass());
    }
    // pm.addPass(toucan::createRemoveSeqPass());
    auto option = toucan::FactorConcatExtractOptions();
    option.parallel = false;
    pm.addPass(toucan::createFactorConcatExtractPass(option));

//    pm.addPass(mlir::createCanonicalizerPass());
    //  pm.addPass(toucan::createParallelCanonicalizerPass());


    if(failed(pm.run(mod.get()))) {
        llvm::outs() << "Passes failed!!!!\n";
        return failure();
    }

    llvm::outs() << "Passes done\n";

    // print output
    auto outputTimer = ts.nest("Write MLIR output");
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(true, true);
    flags.useLocalScope();
    mod->print(output->os());
    output->keep();
    return success();

    // TODO: Output to GPU code

    return success();
}



static LogicalResult toucanMain(MLIRContext &context) {
    mlir::DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    auto ts = tm.getRootScope();


    std::string errorMsg;
    auto input = openInputFile(inputFilename, &errorMsg);

    if (!input) {
        llvm::errs() << errorMsg << "\n";
        return failure();
    }
    auto outputDir = std::filesystem::path(outputDirectory.getValue());


    context.loadDialect<
            hw::HWDialect,
            seq::SeqDialect,
            sv::SVDialect,
            comb::CombDialect,
            om::OMDialect,
            toucan::ToucanDialect
    >();

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    context.printOpOnDiagnostic(true);

    auto result = compileAndEmit(context, ts, sourceMgr, outputDir);

    return result;
}

int main(int argc, char ** argv) {
    llvm::InitLLVM y(argc, argv);
    mlir::MLIRContext context;
    cl::HideUnrelatedOptions(mainCategory);
    toucan::registerPasses();
    mlir::registerCSEPass();
    mlir::registerCanonicalizerPass();
    mlir::registerSCCPPass();
    mlir::registerSymbolDCEPass();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv);
    auto result = toucanMain(context);
    exit(failed(result));
}