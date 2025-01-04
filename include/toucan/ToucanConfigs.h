#pragma once

// Trace all mul ops in generated graph file
#define TOUCAN_DEBUG_TRACE_MUL
// Trace all add ops in generated graph file
#define TOUCAN_DEBUG_TRACE_ADD


#if defined(TOUCAN_DEBUG_TRACE_ADD) || defined(TOUCAN_DEBUG_TRACE_MUL)
#define TOUCAN_DEBUG_TRACE_OP
#endif
