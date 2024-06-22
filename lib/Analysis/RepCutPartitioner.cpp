#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <fstream>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


void RepCutPartitioner::partitionAndSchedule(DesignGraph &graph) {
  auto numVtxes = boost::num_vertices(graph.g);

  levelizeGraph(graph);
  auto cutPoint = findCutPoint(graph, 0.5);
  cutGraph(graph, cutPoint);

  dumpGraphToFile(r1Graph, outputDirectory / "r1.graph");
  dumpGraphToFile(r2Graph, outputDirectory / "r2.graph");

  // TODO
  assert(false && "Not implemented");
}

void RepCutPartitioner::dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const {
  auto ofs = std::ofstream(fileName);

  ofs << boost::num_vertices(g) << ' ' << boost::num_edges(g) << "\n";

  auto numVtxes = boost::num_vertices(g);
  for (uint32_t vtx = 0; vtx < numVtxes; vtx++) {
    ofs << stringifyCGToucanOPName(g[vtx].toucanOpName) << ' ' << g[vtx].weight;
    
    for (auto ei = boost::in_edges(vtx, g); ei.first != ei.second; ++ei.first) {
      auto u = boost::source(*ei.first, g);
      ofs << ' ' << u;
    }
    ofs << "\n";
  }
}
