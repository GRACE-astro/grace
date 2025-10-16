/**
 * @file ml_network_list.cpp
 * @brief ML network storage implementation
 */

#include "grace/ML/ml_networks.hh"

#ifdef GRACE_ENABLE_ML

#include <grace/config/config_parser.hh>
#include <grace/system/print.hh>
#include <grace/errors/assert.hh>
#include <grace/amr/grace_amr.hh>


namespace grace {
namespace ml {

ml_network_list_impl_t::ml_network_list_impl_t()
    : _c2p_network()
{
    GRACE_INFO("Initializing ML network list...");
    
    auto& params = config_parser::get();

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ;
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int batch_size = (nx+2*ngz)*(ny+2*ngz)*(nz+2*ngz)*nq;
    
    bool ml_enabled = false;
    try {
        ml_enabled = params["ml_inference"]["enable_ml"].as<bool>();
    } catch (const std::exception& e) {
        GRACE_INFO("No ML inference configuration found");
        return;
    }
    
    if (!ml_enabled) {
        GRACE_INFO("ML inference disabled");
        return;
    }
    
    try {
        std::string c2p_model_path = params["ml_inference"]["c2p_model_path"].as<std::string>();
        
        if (c2p_model_path.empty()) {
            GRACE_INFO("No C2P model path specified");
            return;
        }
        
        GRACE_INFO("Loading C2P ML model from: {}", c2p_model_path);
        
        // Host-Network temporär erstellen und direkt aus File laden
        ml::MLInferenceNetwork host_network;
        
        // initialize_from_file macht alles: laden + initialisieren
        if (!host_network.initialize_from_file(c2p_model_path, batch_size)) {
            GRACE_WARN("Failed to load C2P model from {}", c2p_model_path);
            return;
        }
        
        // Validierung
        ASSERT(host_network.get_input_size() == 3, 
               "C2P expects 3 inputs, got " << host_network.get_input_size());
        ASSERT(host_network.get_output_size() == 1, 
               "C2P expects 1 output, got " << host_network.get_output_size());
        
        // Log Info
        if (host_network.is_physics_guided()) {
            GRACE_INFO("Physics-guided C2P model with correction scale: {}", 
                      host_network.get_correction_scale());
        } else {
            GRACE_INFO("Standard neural network C2P model");
        }
        
        GRACE_INFO("Network architecture: {}->{}->{}",
                  host_network.get_input_size(),
                  host_network.get_hidden_size(),
                  host_network.get_output_size());
        
        // Device View allokieren und Host-Network kopieren
        _c2p_network = Kokkos::View<ml::MLInferenceNetwork*>("c2p_network", 1);
        auto host_mirror = Kokkos::create_mirror_view(_c2p_network);
        host_mirror(0) = host_network;
        Kokkos::deep_copy(_c2p_network, host_mirror);
        
        GRACE_INFO("C2P network successfully copied to device memory");
        
        // Log optional configuration parameters
        try {
            bool validation_enabled = params["ml_inference"]["c2p_validation_enabled"].as<bool>();
            double fallback_tolerance = params["ml_inference"]["c2p_fallback_tolerance"].as<double>();
            bool log_fallback = params["ml_inference"]["log_fallback_events"].as<bool>();
            
            GRACE_INFO("C2P validation enabled: {}", validation_enabled);
            GRACE_INFO("C2P fallback tolerance: {:.2e}", fallback_tolerance);
            GRACE_INFO("Log fallback events: {}", log_fallback);
        } catch (const std::exception& e) {
            GRACE_INFO("Optional C2P parameters not found, using defaults");
        }
        
    } catch (const std::exception& e) {
        GRACE_WARN("Exception loading C2P: {}", e.what());
    }
    
    GRACE_INFO("ML network list initialization complete");
}

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML