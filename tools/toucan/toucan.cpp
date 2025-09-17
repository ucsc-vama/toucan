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
static cl::opt<bool> gpuCodeGen("gpu", cl::desc("Generate GPU code"), cl::init(false), cl::cat(mainCategory));
// static cl::opt<bool> cpuCodeGen("cpu", cl::desc("Generate CPU code (single thread)"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> dumpOutputMLIR("dump", cl::desc("Dump output.mlir"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> removeSVDialect("removeSVDialect", cl::desc("Remove print and assertion"), cl::init(false), cl::cat(mainCategory));
// static cl::opt<uint32_t> targetGPUSMs("gpuSMs", cl::desc("Target GPU SM count. This argument affects number of partitions if partSize is not explicitely given."), cl::init(142), cl::cat(mainCategory));
// static cl::opt<uint32_t> partitionNumRegions("numRegions", cl::desc("Partitioning region count."), cl::init(4), cl::cat(mainCategory));
static cl::opt<float> partitionIbFactor("ibFactor", cl::desc("Partitioning target imbalance factor (1.0 > ibFactor > 0.0)."), cl::init(0.015), cl::cat(mainCategory));
static cl::opt<float> partitionSizeRatio("partSize", cl::desc("Force partition size ration between 0.1 and 1.0. Values outside of this range would allow toucan decide part size automatically"), cl::init(1.0), cl::cat(mainCategory));

void checkArgs() {
    // if (!cpuCodeGen && !gpuCodeGen) {
    //     llvm::outs() << "Please specify at least one code gen type (-cpu or -gpu)\n";
    //     exit(-1);
    // }
}

static LogicalResult compileAndEmit(
        MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr, std::filesystem::path &outputDir) {

    std::string errorMsg;

    auto mod = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if(!mod) return failure();

    mlir::PassManager pm(&context);
    if(failed(applyPassManagerCLOptions(pm)))
        return failure();

    pm.enableTiming(ts);

    if (inputLevel < ToucanHigh) {
        // Lower to ToucanHigh

        if (!removeSVDialect) {
            // Expand SV macros, generates Print and Stop
            pm.addPass(toucan::createExpandSVMacroPass());
        }
        // Remove unsupported Ops (other SV Ops, OM Ops)
        pm.addPass(toucan::createRemoveSVnOMPass());
        // SplitRWPort
        pm.addPass(toucan::createSplitFirMemRWPortsPass());
        // Expand memory delays
        pm.addPass(toucan::createExpandMemoryDelayPass());
        // // Remove mem read en signal
        // pm.addPass(toucan::createRemoveMemReadEnPass());
        // Remove all mem write masks. Split memory if there is any mask
        pm.addPass(toucan::createRemoveMemMaskPass());

        // Split registers into def, read and write nodes
        pm.addPass(toucan::createSplitRegistersPass());

        // pm.addPass(toucan::createCanonicalizerPass());
    }

    if (inputLevel < Toucan4B) {
        // Lower to Toucan4B
        // Lower registers and memory to 4b
        pm.addPass(toucan::createLowerRegMemTo4BPass());
        // By this time clock should be removed by DCE. AsClock is not supported
        pm.addPass(toucan::createEnsureNoClockOpPass());

        pm.addPass(toucan::createFactorHWMiscPass());

        // Lower HW vector
        pm.addPass(toucan::createFactorArrayGetMuxPass());
        pm.addPass(toucan::createLowerHWVectorTo4BPass());
        // Fold binary ops with more than 2 operands
        pm.addPass(toucan::createFactorBinaryOpPass());
        // Lower Comb to 4B
        pm.addPass(toucan::createLowerCombTo4B_1Pass());
        pm.addPass(toucan::createLowerCombTo4B_2Pass());
        pm.addPass(toucan::createLowerCombTo4B_3Pass());

        // FactorConcatExtract per module (for compilation speedup)
        pm.addPass(toucan::createFactorConcatExtractPass());
        pm.addPass(toucan::createToucanCanonicalizerPass());
    }

    if (inputLevel < ToucanFlattened) {
        // Lower to flattened
        pm.addPass(toucan::createFlattenPass());
        pm.addPass(toucan::createFactorConcatExtractPass());
        // Canonicalizer also works on flatten design.
        // Currently no benefits for canonicalization after flatten
        pm.addPass(toucan::createRemoveConstRegsPass());
        pm.addPass(toucan::createToucanCanonicalizerPass());
        pm.addPass(toucan::createDeduplicateRegistersPass());
        // pm.addPass(toucan::createRemoveConstRegsPass());
        pm.addPass(toucan::createMergeConstPass());
        pm.addPass(toucan::createEnsureToucanOnlyPass());
        pm.addPass(toucan::createBreakPinnedValueToOutputConnectionPass());
    }

    // if (cpuCodeGen) {
    //     auto cpuCodeGenOptions = toucan::CPUSingleThreadCodeGenOptions();
    //     cpuCodeGenOptions.outputDirectory = outputDir.string();
    //     cpuCodeGenOptions.outputDesignFilename = "CPUSimDesign.bin";
    //     cpuCodeGenOptions.outputSymbolFilename = "CPUSimSymbols.bin";
    //     cpuCodeGenOptions.outputIOSymbolFilename = "CPUSimIOSymbols.bin";
    //     cpuCodeGenOptions.temporaryDirectory = outputDir.string();

    //     pm.addPass(toucan::createCPUSingleThreadCodeGenPass(cpuCodeGenOptions));
    //     llvm::outs() << "CPU Code gen\n";
    // }

    if (gpuCodeGen) {
        auto gpuCodeGenOptions = toucan::GPUCodeGenOptions();
        gpuCodeGenOptions.outputDirectory = outputDir.string();
        gpuCodeGenOptions.outputDesignFilename = "GPUSimDesign.bin";
        gpuCodeGenOptions.outputSymbolFilename = "GPUSimSymbols.bin";
        gpuCodeGenOptions.outputIOSymbolFilename = "GPUSimIOSymbols.bin";
        gpuCodeGenOptions.temporaryDirectory = outputDir.string();
        // gpuCodeGenOptions.numRegions = partitionNumRegions;
        gpuCodeGenOptions.ibFactor = partitionIbFactor;
        gpuCodeGenOptions.partSizeRatio = partitionSizeRatio;
        // gpuCodeGenOptions.targetSMs = targetGPUSMs;

        pm.addPass(toucan::createGPUCodeGenPass(gpuCodeGenOptions));
    }



    if(failed(pm.run(mod.get()))) {
        llvm::outs() << "Passes failed!!!!\n";
        return failure();
    }

    llvm::outs() << "Passes done\n";

    // print output
    if (dumpOutputMLIR) {
        auto outputTimer = ts.nest("Write MLIR output");
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(true, true);
        flags.useLocalScope();

        auto outputFilename = outputDir / "output.mlir";
        auto output = openOutputFile(outputFilename.string(), &errorMsg);
        if (!output) {
            llvm::errs() << errorMsg << "\n";
            return failure();
        }
        mod->print(output->os());
        output->keep();
    }

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
    // context.disableMultithreading();
    cl::HideUnrelatedOptions(mainCategory);
    toucan::registerPasses();
    mlir::registerCSEPass();
    // mlir::registerCanonicalizerPass();
    mlir::registerSCCPPass();
    mlir::registerSymbolDCEPass();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv);
    checkArgs();
    auto result = toucanMain(context);
    exit(failed(result));
}