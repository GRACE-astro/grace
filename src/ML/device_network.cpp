#include "grace/ML/device_network.hh"
#include "grace/ML/model_loader.hh"

#ifdef GRACE_ENABLE_ML

#include "grace/system/print.hh"
#include <spdlog/spdlog.h>

namespace grace {
namespace ml {

bool MLInferenceNetwork::initialize_from_file(const std::string& model_file, int batch_size_) {
    try {
        NetworkConfig config;
        HostMatrix2D h_weights_ih, h_weights_ho;
        HostVector1D h_bias_h, h_bias_o;

        // Load model from HDF5
        if (!ModelLoader::load_model_from_hdf5(model_file, config,
                                              h_weights_ih, h_bias_h,
                                              h_weights_ho, h_bias_o)) {
            spdlog::error("Failed to load model from file: {}", model_file);
            return false;
        }

        // Set network dimensions
        input_size_ = config.input_size;
        hidden_size_ = config.hidden_size;
        output_size_ = config.output_size;
        has_normalization_ = config.has_normalization;
        is_physics_guided_ = config.is_physics_guided;
        correction_scale_ = config.correction_scale;


        // Allocate device memory
        weights_input_to_hidden_ = Matrix2D("weights_ih", input_size_, hidden_size_);
        bias_hidden_ = Vector1D("bias_h", hidden_size_);
        weights_hidden_to_output_ = Matrix2D("weights_ho",hidden_size_, output_size_);
        bias_output_ = Vector1D("bias_o", output_size_);

        // Copy weights from host to device
        Kokkos::deep_copy(bias_hidden_, h_bias_h);
        Kokkos::deep_copy(bias_output_, h_bias_o);

        // Allocate device memory
        auto h_weights_input_to_hidden_trans_ = HostMatrix2D("weights_ih", input_size_, hidden_size_);
        auto h_weights_hidden_to_output_trans_ = HostMatrix2D("weights_ho", hidden_size_, output_size_);

        // Transpose on host (normal for-loops)
        for (int i = 0; i < input_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
                h_weights_input_to_hidden_trans_(i, j) = h_weights_ih(j, i);
            }
        }

        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < output_size_; j++) {
                h_weights_hidden_to_output_trans_(i, j) = h_weights_ho(j, i);
            }
        }

        // Copy weights from host to device
        Kokkos::deep_copy(weights_input_to_hidden_, h_weights_input_to_hidden_trans_);
        Kokkos::deep_copy(weights_hidden_to_output_, h_weights_hidden_to_output_trans_);


        GRACE_INFO("Allocating batch processing buffers (batch_size={})...", batch_size_);
        
        // Allocate batch processing buffers
        // In initialize_from_file, add:
        batch_inputs_ = Matrix2D("batch_inputs", batch_size_, input_size_);
        batch_hidden_pre_activation_ = Matrix2D("batch_hidden_pre", batch_size_, hidden_size_);
        batch_hidden_post_activation_ = Matrix2D("batch_hidden_post", batch_size_, hidden_size_);
        batch_output_pre_activation_ = Matrix2D("batch_output_pre", batch_size_, output_size_);
        batch_output_ = Matrix2D("batch_output", batch_size_, output_size_);

        // Calculate and log memory usage
        size_t weights_memory = (hidden_size_ * input_size_ + hidden_size_ + 
                                output_size_ * hidden_size_ + output_size_) * sizeof(real_t);
        size_t batch_memory = batch_size_ * (2 * hidden_size_ + output_size_) * sizeof(real_t);
        size_t total_memory = weights_memory + batch_memory;
        
        GRACE_INFO("Network memory usage:");
        GRACE_INFO("  Weights: {:.2f} MB", weights_memory / 1024.0 / 1024.0);
        GRACE_INFO("  Batch buffers: {:.2f} MB", batch_memory / 1024.0 / 1024.0);
        GRACE_INFO("  Total: {:.2f} MB", total_memory / 1024.0 / 1024.0);

        // Handle normalization parameters
        if (has_normalization_) {
            input_min_ = Vector1D("input_min", input_size_);
            input_max_ = Vector1D("input_max", input_size_);
            Kokkos::deep_copy(input_min_, config.input_min);
            Kokkos::deep_copy(input_max_, config.input_max);

            output_min_ = Vector1D("output_min", output_size_);  
            output_max_ = Vector1D("output_max", output_size_);  
            Kokkos::deep_copy(output_min_, config.output_min);  
            Kokkos::deep_copy(output_max_, config.output_max);  
        }

        is_ready_ = true;

        if (is_physics_guided_) {
            GRACE_INFO("Physics-guided ML network initialized: {}x{}x{}, correction_scale: {}",
                      input_size_, hidden_size_, output_size_, correction_scale_);
        } else {
            GRACE_INFO("Standard ML network initialized: {}x{}x{}",
                      input_size_, hidden_size_, output_size_);
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize ML network from file: {}", e.what());
        is_ready_ = false;
        return false;
    }
}


bool MLInferenceNetwork::initialize_from_config(const NetworkConfig& config) {
    try {
        // Set network dimensions
        input_size_ = config.input_size;
        hidden_size_ = config.hidden_size;
        output_size_ = config.output_size;
        has_normalization_ = config.has_normalization;
        is_physics_guided_ = config.is_physics_guided;
        correction_scale_ = config.correction_scale;

        // Allocate device memory
        weights_input_to_hidden_ = Matrix2D("weights_ih", hidden_size_, input_size_);
        bias_hidden_ = Vector1D("bias_h", hidden_size_);
        weights_hidden_to_output_ = Matrix2D("weights_ho", output_size_, hidden_size_);
        bias_output_ = Vector1D("bias_o", output_size_);

        // Handle normalization parameters
        if (has_normalization_) {
            input_min_ = Vector1D("input_min", input_size_);
            input_max_ = Vector1D("input_max", input_size_);
            Kokkos::deep_copy(input_min_, config.input_min);
            Kokkos::deep_copy(input_max_, config.input_max);
        }

        // Initialize with zero weights if no file provided
        GRACE_WARN("Network initialized with zero weights - load from file for actual model");
        is_ready_ = true;
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize ML network from config: {}", e.what());
        is_ready_ = false;
        return false;
    }
}

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML