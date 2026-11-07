// Host-only regression: build with --target check-design-graph, then run it.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace toucan;

int main() {
  MLIRContext context;
  context.getOrLoadDialect<ToucanDialect>();
  // Repeated operands and dependencies between merge groups must remain
  // parallel edges. Static segment reads contribute no extra operation count.
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  %c = toucan.const 0 : i4 : i4
  %en = toucan.const true : i1
  %reg = toucan.def_reg : reg i4
  %rr = toucan.reg_read %reg : reg i4 -> i4
  %v = toucan.vector %rr, %c, %c : i4, i4, i4 -> vec i4[3] false
  %r0 = toucan.vec_read %v[i1 %en + 0 : i16] out_range i4 %c : vec i4[3] false -> i4
  %r1 = toucan.vec_read %v[i1 %en + 1 : i16] out_range i4 %c : vec i4[3] false -> i4
  %v2 = toucan.vector %r0, %r1, %r0, %r1, %r0, %r1, %r0, %r1 : i4, i4, i4, i4, i4, i4, i4, i4 -> vec i4[8] false
  %a = toucan.vec_arith vaadd %v2 %v2 : vec i4[8] false vec i4[8] false -> vec i4[8] false
  %s0 = toucan.vec_read_segment %a segment 0 : i16 : vec i4[8] false -> i4
  %s1 = toucan.vec_read_segment %a segment 1 : i16 : vec i4[8] false -> i4
  %mem = toucan.def_mem : mem i4[16]
  %mr = toucan.mem_read %mem[vec i4[8] false %v2] en %en : mem i4[16] -> i4
  toucan.mem_write %mem[%v2] = %s0 en %en : mem i4[16][vec i4[8] false] i4 en i1 -> ()
  toucan.mem_write %mem[%v2] = %s1 en %en : mem i4[16][vec i4[8] false] i4 en i1 -> ()
  toucan.reg_write %reg %mr : (reg i4 i4)
  %unused_mem = toucan.def_mem : mem i4[16]
  %cv = toucan.const_vector [0 : i4] : vec i4[1] true
  %single = toucan.vec_read %cv[i1 %en + 0 : i16] out_range i4 %c : vec i4[1] true -> i4
  %single_mem = toucan.def_mem : mem i4[16]
  toucan.mem_write %single_mem[%v2] = %single en %en : mem i4[16][vec i4[8] false] i4 en i1 -> ()
}
)mlir",
                                            &context);
  if (!module)
    return 1;

  std::vector<Operation *> ops;
  for (auto &op : module->getOps())
    ops.push_back(&op);
  ModuleAnalysisManager mam(*module, nullptr);
  AnalysisManager am = mam;
  DesignGraph graph(*module, am);

  struct ExpectedVertex {
    unsigned op;
    CGToucanOPName kind;
    unsigned weight;
    unsigned count;
    std::vector<unsigned> users;
  };
  using K = CGToucanOPName;
  // Operation indexes identify the retained representative, independent of
  // graph numbering. User lists are multisets, including repeated edges.
  std::vector<ExpectedVertex> expected = {
      {0, K::ConstDecl, 0, 1, {4, 4, 5, 5, 18}},
      {1, K::ConstDecl, 0, 1, {5, 5, 12, 13, 13, 18, 20}},
      {3, K::RegRead, 10, 1, {4}},
      {4, K::VecDecl, 21, 3, {5, 5}},
      {5, K::VecRead, 10, 2, {7, 7, 7, 7, 7, 7, 7, 7}},
      {7, K::VecDecl, 56, 8, {8, 8, 12, 13, 13, 20}},
      {8, K::VecArith, 80, 1, {13, 13}},
      {12, K::MemRead, 200, 1, {15}},
      {13, K::MemWrite, 200, 2, {}},
      {15, K::RegWrite, 5, 1, {}},
      {17, K::ConstDecl, 0, 1, {18}},
      {18, K::VecRead, 10, 1, {20}},
      {20, K::MemWrite, 200, 1, {}},
  };
  auto fail = [](const char *message) {
    llvm::errs() << "FAIL: " << message << '\n';
    return 1;
  };
  if (boost::num_vertices(graph.g) != expected.size() || graph.opToId.size() != expected.size())
    return fail("vertex count or operation map");
  if (graph.regs.size() != 1 || !graph.regs.contains(cast<DefRegOp>(ops[2]).getHandle()))
    return fail("register set");

  std::map<Operation *, unsigned> opIndexes;
  for (unsigned i = 0; i < ops.size(); ++i)
    opIndexes[ops[i]] = i;
  for (unsigned i = 0; i < expected.size(); ++i) {
    auto &e = expected[i];
    const auto &v = graph.g[i];
    if (v.op != ops[e.op] || v.toucanOpName != e.kind || v.weight != e.weight || v.opCount != e.count ||
        v.exchangeValId != 0 || v.mp || !graph.opToId.count(v.op) || graph.opToId.lookup(v.op) != i)
      return fail("vertex metadata or representative order");
    std::vector<unsigned> users;
    for (auto edge : boost::make_iterator_range(boost::out_edges(i, graph.g)))
      users.push_back(opIndexes.at(graph.g[boost::target(edge, graph.g)].op));
    std::sort(users.begin(), users.end());
    std::sort(e.users.begin(), e.users.end());
    if (users != e.users) {
      llvm::errs() << "Operation index " << e.op << '\n';
      return fail("dependency multiset");
    }
  }
  // Graph regions can represent cycles. Construction must not silently erase
  // an existing self-edge on a retained vertex (later scheduling may reject it).
  auto cyclic = parseSourceString<ModuleOp>(R"mlir(
module {
  %c = toucan.const 0 : i4 : i4
  %en = toucan.const true : i1
  %v = toucan.vector %c : i4 -> vec i4[1] false
  %r = toucan.vec_read %v[i1 %en + 0 : i16] out_range i4 %r : vec i4[1] false -> i4
}
)mlir",
                                            &context);
  if (!cyclic)
    return 1;
  ModuleAnalysisManager cyclicMAM(*cyclic, nullptr);
  AnalysisManager cyclicAM = cyclicMAM;
  DesignGraph cyclicGraph(*cyclic, cyclicAM);
  if (boost::num_vertices(cyclicGraph.g) != 4 || boost::num_edges(cyclicGraph.g) != 4 ||
      !boost::edge(3, 3, cyclicGraph.g).second)
    return fail("existing self-edge was lost");

  llvm::outs() << "PASS: representatives, metadata, registers, dependency multiplicities, and self-edges\n";
  return 0;
}
