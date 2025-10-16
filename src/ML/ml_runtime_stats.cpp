/**
 * @file ml_runtime_stats.cpp
 * @brief ML inference statistics and configuration helpers
 */

#include "grace/ML/ml_runtime.hh"

#ifdef GRACE_ENABLE_ML

#include "grace/config/config_parser.hh"
#include "grace/system/print.hh"
#include "grace/ML/ml_networks.hh"

namespace grace {
namespace ml_runtime {

namespace {
    bool is_initialized_ = false;
}


void initialize() {
    if (is_initialized_) {
        GRACE_INFO("ML runtime already initialized, skipping");
        return;
    }
    
    GRACE_INFO("Initializing ML inference system...");
    
    try {
        // EXPLIZIT initialisieren statt implizit via get()
        ml::ml_network_list::initialize();  // ← NEU!
        // Trigger singleton creation - this loads all models
        auto& networks = ml::ml_network_list::get();
        
        // Log what's available
        if (networks.is_c2p_initialized()) {
            GRACE_INFO("C2P ML network ready for inference");
        } else {
            GRACE_INFO("C2P ML network not available (disabled or failed to load)");
        }
        
        // Log configuration
        if (networks.is_c2p_initialized()) {
            try {
                auto& params = config_parser::get();
                bool validation = params["ml_inference"]["c2p_validation_enabled"].as<bool>();
                double tolerance = params["ml_inference"]["c2p_fallback_tolerance"].as<double>();
                bool log_fallback = params["ml_inference"]["log_fallback_events"].as<bool>();
                
                GRACE_INFO("ML Configuration:");
                GRACE_INFO("  Validation enabled: {}", validation);
                GRACE_INFO("  Fallback tolerance: {:.2e}", tolerance);
                GRACE_INFO("  Log fallback events: {}", log_fallback);
            } catch (const std::exception& e) {
                GRACE_INFO("Using default ML configuration parameters");
            }
        }
        
        is_initialized_ = true;
        GRACE_INFO("ML inference system initialization complete");
        
    } catch (const std::exception& e) {
        GRACE_WARN("Failed to initialize ML inference system: {}", e.what());
        GRACE_WARN("Continuing without ML inference");
    }
}

void finalize() {
    if (!is_initialized_) {
        return;
    }
    
    GRACE_INFO("Finalizing ML inference system...");
    
    // Log statistics if enabled
    log_inference_statistics();
    
    // Kokkos Views handle their own cleanup
    // Singleton will be destroyed automatically
    
    is_initialized_ = false;
    GRACE_INFO("ML inference system finalized");
}


// ========== Statistics Implementation ==========

namespace {
    std::atomic<size_t> total_inferences{0};
    std::atomic<size_t> successful_inferences{0};
    std::atomic<size_t> fallback_events{0};
}

void increment_inference_stats(bool success, bool used_fallback) {
    total_inferences++;
    if (success) {
        successful_inferences++;
    }
    if (used_fallback) {
        fallback_events++;
    }
}

void log_inference_statistics() {
    if (!get_log_inference_stats()) {
        return;
    }
    
    size_t total = total_inferences.load();
    size_t successful = successful_inferences.load();
    size_t fallbacks = fallback_events.load();
    
    if (total > 0) {
        double success_rate = 100.0 * successful / total;
        double fallback_rate = 100.0 * fallbacks / total;
        
        GRACE_INFO("ML Inference Statistics:");
        GRACE_INFO("  Total inferences: {}", total);
        GRACE_INFO("  Success rate: {:.2f}%", success_rate);
        GRACE_INFO("  Fallback rate: {:.2f}%", fallback_rate);
    }
}

void reset_inference_statistics() {
    total_inferences = 0;
    successful_inferences = 0;
    fallback_events = 0;
}

// ========== Configuration Helpers Implementation ==========

bool get_c2p_validation_enabled() {
    try {
        return grace::get_param<bool>("ml_inference", "c2p_validation_enabled");
    } catch (const std::exception& e) {
        return true; // Default to validation enabled
    }
}

double get_c2p_fallback_tolerance() {
    try {
        return grace::get_param<double>("ml_inference", "c2p_fallback_tolerance");
    } catch (const std::exception& e) {
        return 1e-10; // Default tolerance
    }
}

bool get_log_fallback_events() {
    try {
        return grace::get_param<bool>("ml_inference", "log_fallback_events");
    } catch (const std::exception& e) {
        return true; // Default to logging enabled
    }
}

bool get_log_inference_stats() {
    try {
        return grace::get_param<bool>("ml_inference", "log_inference_stats");
    } catch (const std::exception& e) {
        return false; // Default to no stats logging
    }
}

} // namespace ml_runtime
} // namespace grace

#endif // GRACE_ENABLE_ML