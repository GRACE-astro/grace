#ifndef GRACE_ML_INFERENCE_ML_RUNTIME_HH
#define GRACE_ML_INFERENCE_ML_RUNTIME_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_ML

#include <string>
#include <atomic>

namespace grace {
namespace ml_runtime {

// ========== Statistics Tracking ==========

/**
 * @brief Increment inference statistics
 * @param success Whether the inference was successful
 * @param used_fallback Whether fallback was used
 */
void increment_inference_stats(bool success, bool used_fallback);

/**
 * @brief Log current inference statistics
 */
void log_inference_statistics();

/**
 * @brief Reset inference statistics counters
 */
void reset_inference_statistics();

// ========== Configuration Helpers ==========

/**
 * @brief Get C2P validation enabled setting
 * @return true if validation is enabled (default: true)
 */
bool get_c2p_validation_enabled();

/**
 * @brief Get C2P fallback tolerance setting
 * @return fallback tolerance value (default: 1e-10)
 */
double get_c2p_fallback_tolerance();

/**
 * @brief Get log fallback events setting
 * @return true if fallback events should be logged (default: true)
 */
bool get_log_fallback_events();

/**
 * @brief Get log inference stats setting
 * @return true if inference statistics should be logged (default: false)
 */
bool get_log_inference_stats();

/**
 * @brief Initialize ML inference system
 * 
 * Loads all configured ML models and prepares them for inference.
 * Must be called after Kokkos initialization and config parser initialization.
 * 
 * This function:
 * - Triggers singleton creation of ml_network_list
 * - Loads models from HDF5 files
 * - Copies networks to device memory
 * - Validates network architectures
 * 
 * Safe to call multiple times (no-op after first call).
 */
void initialize();

/**
 * @brief Finalize ML inference system
 * 
 * Logs statistics and performs cleanup.
 * Should be called during GRACE shutdown.
 */
void finalize();


} // namespace ml_runtime
} // namespace grace

#endif // GRACE_ENABLE_ML
#endif // GRACE_ML_INFERENCE_ML_RUNTIME_HH