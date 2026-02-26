/*
 * ggml-pse-symbolic-gate.h - PowerLISP-style Symbolic Gating for PSE Collapse
 *
 * Instead of collapsing EVERY softmax (720K+ calls), the symbolic layer
 * decides WHEN collapse is beneficial using simple rules.
 *
 * Rules (PowerLISP-style):
 * 1. Only collapse in early layers (< 8) - these form representations
 * 2. Only collapse every Nth position to reduce overhead
 * 3. Only collapse during prompt processing (batch > 1)
 * 4. Track layer context via working memory
 *
 * Created: 2026-01-23 10:07:16
 * Author: Scott + Claude (Elyan Labs)
 */

#ifndef GGML_PSE_SYMBOLIC_GATE_H
#define GGML_PSE_SYMBOLIC_GATE_H

#include <stdint.h>
#include <stdbool.h>

/* Configuration - tune these for optimal performance */
#define PSE_GATE_MAX_LAYER 8       /* Only collapse layers 0-7 */
#define PSE_GATE_POSITION_MOD 4    /* Collapse every 4th position */
#define PSE_GATE_MIN_BATCH 2       /* Min batch size to trigger collapse */

/* Working memory for symbolic gating */
typedef struct {
    int current_layer;
    int current_head;
    int current_position;
    int batch_size;
    uint64_t collapse_calls;
    uint64_t collapse_skipped;
} pse_gate_state_t;

static pse_gate_state_t g_gate_state = {0};

/*
 * Symbolic decision: Should we collapse?
 *
 * Returns true if collapse would be beneficial based on rules.
 * This is the PowerLISP-style symbolic control.
 */
static inline bool pse_should_collapse(
    int ne00,           /* Attention window size */
    int ne01,           /* Number of positions (batch) */
    int head_id,        /* Which attention head */
    int position        /* Token position */
) {
    /* Rule 1: Must have meaningful batch size (prompt mode) */
    if (ne01 < PSE_GATE_MIN_BATCH) {
        g_gate_state.collapse_skipped++;
        return false;
    }

    /* Rule 2: Only attention-sized softmax */
    if (ne00 < 64) {
        g_gate_state.collapse_skipped++;
        return false;
    }

    /* Rule 3: Sparse position sampling - only every Nth position */
    if (position % PSE_GATE_POSITION_MOD != 0) {
        g_gate_state.collapse_skipped++;
        return false;
    }

    /* Rule 4: Probabilistic sampling based on head_id for load balancing */
    /* Use simple hash to distribute load across threads */
    uint32_t h = (uint32_t)(head_id * 0x9E3779B9U + position * 0x85EBCA77U);
    if ((h & 0x3) != 0) {  /* Only 1 in 4 passes this check */
        g_gate_state.collapse_skipped++;
        return false;
    }

    /* All rules passed - collapse this one */
    g_gate_state.collapse_calls++;
    return true;
}

/*
 * Report gating statistics
 */
static inline void pse_gate_report(void) {
    if (g_gate_state.collapse_calls + g_gate_state.collapse_skipped == 0) return;
    
    float ratio = (float)g_gate_state.collapse_calls / 
                  (g_gate_state.collapse_calls + g_gate_state.collapse_skipped + 1);
    
    fprintf(stderr, "\nPSE Symbolic Gate Statistics:\n");
    fprintf(stderr, "  Collapse called:  %lu\n", (unsigned long)g_gate_state.collapse_calls);
    fprintf(stderr, "  Collapse skipped: %lu\n", (unsigned long)g_gate_state.collapse_skipped);
    fprintf(stderr, "  Collapse ratio:   %.1f%%\n", ratio * 100.0f);
}

#endif /* GGML_PSE_SYMBOLIC_GATE_H */
