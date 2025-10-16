#ifndef GRACE_ML_INFERENCE_NETWORK_HH
#define GRACE_ML_INFERENCE_NETWORK_HH

#include <Kokkos_Core.hpp>
#include <grace_config.h>

#ifdef GRACE_ENABLE_ML

#include <string>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>
#include "grace/ML/model_loader.hh"
#include "grace/ML/network_config.hh"
#include <grace/amr/grace_amr.hh>

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

namespace grace {

#if defined(GRACE_ENABLE_CUDA)
using default_space_ai = Kokkos::CudaSpace   ;  
#elif defined(GRACE_ENABLE_HIP)
using default_space_ai = Kokkos::HIPSpace    ;
#elif defined(GRACE_ENABLE_OMP) or defined(GRACE_ENABLE_SERIAL)
using default_space_ai = Kokkos::HostSpace   ;
#endif   

namespace ml {
/**
 * @brief Unified ML inference network that works on both host and device
 * Similar to tabulated_eos_t pattern - single class with Kokkos Views
 */
class MLInferenceNetwork {
public:
    // Default constructor
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    MLInferenceNetwork() = default;
    
    // Constructor with configuration
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    MLInferenceNetwork(
        Matrix2D weights_input_to_hidden,
        Vector1D bias_hidden,
        Matrix2D weights_hidden_to_output,
        Vector1D bias_output,
        Vector1D input_min,
        Vector1D input_max,
        int input_size,
        int hidden_size,
        int output_size,
        bool has_normalization,
        bool is_physics_guided,
        real_t correction_scale
    ) : weights_input_to_hidden_(weights_input_to_hidden)
      , bias_hidden_(bias_hidden)
      , weights_hidden_to_output_(weights_hidden_to_output)
      , bias_output_(bias_output)
      , input_min_(input_min)
      , input_max_(input_max)
      , input_size_(input_size)
      , hidden_size_(hidden_size)
      , output_size_(output_size)
      , has_normalization_(has_normalization)
      , is_physics_guided_(is_physics_guided)
      , correction_scale_(correction_scale)
      , is_ready_(true)
    {}

    // Copy constructor
    MLInferenceNetwork(const MLInferenceNetwork& other)
        : weights_input_to_hidden_(other.weights_input_to_hidden_)
        , bias_hidden_(other.bias_hidden_)
        , weights_hidden_to_output_(other.weights_hidden_to_output_)
        , bias_output_(other.bias_output_)
        , input_min_(other.input_min_)
        , input_max_(other.input_max_)
        , input_size_(other.input_size_)
        , output_min_(other.output_min_)
        , output_max_(other.output_max_)
        , hidden_size_(other.hidden_size_)
        , output_size_(other.output_size_)
        , has_normalization_(other.has_normalization_)
        , is_physics_guided_(other.is_physics_guided_)
        , correction_scale_(other.correction_scale_)
        , is_ready_(other.is_ready_)
        , batch_inputs_(other.batch_inputs_)
        , batch_output_(other.batch_output_)
        , batch_hidden_pre_activation_(other.batch_hidden_pre_activation_)
        , batch_hidden_post_activation_(other.batch_hidden_post_activation_)
        , batch_output_pre_activation_(other.batch_output_pre_activation_)
    {}

    // Initialize from file (host-only function)
    bool initialize_from_file(const std::string& model_file, int batch_size_);
    
    // Initialize from configuration (host-only function)
    bool initialize_from_config(const NetworkConfig& config);

    /**
     * @brief Device-safe forward inference
     * This function can be called from both host and device code
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void forward(const real_t* input, real_t* output) const {
        if (!is_ready_) {
            output[0] = 1.0; // Safe fallback
            return;
        }

        // Normalize input if needed
        real_t normalized_input[8]; // Assuming max 8 inputs
        if (has_normalization_) {
            for (int i = 0; i < input_size_; ++i) {
                normalized_input[i] = (input[i] - input_min_(i)) / (input_max_(i) - input_min_(i));
            }
        } else {
            for (int i = 0; i < input_size_; ++i) {
                normalized_input[i] = input[i];
            }
        }
        
        if (is_physics_guided_) {
            // Physics-guided approach: analytical_guess + correction
            real_t z_baseline = analytical_guess(normalized_input[0], normalized_input[1], normalized_input[2]);
            
            // Get correction from neural network
            real_t correction = neural_network_forward(normalized_input);
            
            // Apply correction with scale
            output[0] = z_baseline + correction_scale_ * correction;
        } else {
            // Standard neural network forward pass
            output[0] = neural_network_forward(normalized_input);
        }

        // Denormalize output if needed
        if (has_normalization_) {
            output[0] = output[0] * (output_max_(0) - output_min_(0)) + output_min_(0);
        } else {
            output[0] = output[0];
        }
    }

    /**
     * @brief Batched forward inference for multiple inputs
     * Uses pre-allocated buffers and KokkosBlas for efficient matrix operations
     * 
     * @param inputs Input matrix [batch_size, input_size]
     * @param outputs Output matrix [batch_size, output_size]  
     * @param batch_size Number of samples to process (must be <= initialized batch_size)
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void batched_forward(
        int batch_size
    ) const {
        if (!is_ready_) return;

        // Validate batch size doesn't exceed pre-allocated buffers
        //if (batch_size > batch_hidden_pre_activation_.extent(0)) {
        //    // Fallback to single inference if batch too large
        //    Kokkos::parallel_for("fallback_forward", batch_size, 
        //        KOKKOS_LAMBDA(const int i) {
        //            real_t input_arr[8];
        //            real_t output_arr[1];
        //            for (int j = 0; j < input_size_; ++j) {
        //                input_arr[j] = batch_inputs_(i, j);
        //            }
        //            forward(input_arr, output_arr);
        //            batch_output_(i, 0) = output_arr[0];
        //        });
        //    return;
        //}

        // Create subviews for the actual batch size being processed
        auto input = Kokkos::subview(batch_inputs_, 
                                          Kokkos::make_pair(0, batch_size), 
                                          Kokkos::ALL());
        auto hidden_pre = Kokkos::subview(batch_hidden_pre_activation_, 
                                          Kokkos::make_pair(0, batch_size), 
                                          Kokkos::ALL());
        auto hidden_post = Kokkos::subview(batch_hidden_post_activation_, 
                                           Kokkos::make_pair(0, batch_size), 
                                           Kokkos::ALL());
        auto output_pre = Kokkos::subview(batch_output_pre_activation_, 
                                          Kokkos::make_pair(0, batch_size), 
                                          Kokkos::ALL());
        auto outputs = Kokkos::subview(batch_output_, 
                                          Kokkos::make_pair(0, batch_size), 
                                          Kokkos::ALL());

        // Step 1: Input → Hidden (pre-activation)
        // hidden_pre[batch, hidden] = inputs[batch, input] * weights_ih[hidden, input]^T
        KokkosBlas::gemm("N", "N", 1.0, input, weights_input_to_hidden_, 0.0, hidden_pre);
        //KokkosBlas::gemm("N", "T", 1.0, input, weights_input_to_hidden_, 0.0, hidden_pre);

        // Step 2: Add bias and apply activation → Hidden (post-activation)
        Kokkos::parallel_for("hidden_activation",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {batch_size, hidden_size_}),
            [=,this] GRACE_HOST_DEVICE(const int i, const int h) {
                real_t x = hidden_pre(i, h) + bias_hidden_(h);
                hidden_post(i, h) = Kokkos::tanh(x);
            });

        // Step 3: Hidden → Output (pre-activation)
        // output_pre[batch, output] = hidden_post[batch, hidden] * weights_ho[output, hidden]^T
        KokkosBlas::gemm("N", "N", 1.0, hidden_post, weights_hidden_to_output_, 0.0, output_pre);
        //KokkosBlas::gemm("N", "T", 1.0, hidden_post, weights_hidden_to_output_, 0.0, output_pre);

        // Step 4: Add bias, apply physics-guided correction, and denormalize
        if (is_physics_guided_) {
            Kokkos::parallel_for("output_physics_guided", batch_size,
                [=,this] GRACE_HOST_DEVICE(const int i) {
                    // Add bias
                    real_t nn_output = output_pre(i, 0) + bias_output_(0);

                    // Get analytical baseline (using normalized inputs)
                    real_t D = input(i, 0);
                    real_t q = input(i, 1);
                    real_t r = input(i, 2);
                    real_t z_baseline = analytical_guess(D, q, r);

                    // Apply correction
                    real_t corrected_output = z_baseline + correction_scale_ * nn_output;

                    // Denormalize if needed
                    if (has_normalization_) {
                        outputs(i, 0) = corrected_output * (output_max_(0) - output_min_(0)) + 
                                       output_min_(0);
                    } else {
                        outputs(i, 0) = corrected_output;
                    }
                });
        } else {
            // Standard neural network output
            Kokkos::parallel_for("output_standard", batch_size,
                [=,this] GRACE_HOST_DEVICE(const int i) {
                    // Add bias
                    real_t nn_output = output_pre(i, 0) + bias_output_(0);

                    // Denormalize if needed
                    if (has_normalization_) {
                        outputs(i, 0) = nn_output * (output_max_(0) - output_min_(0)) + 
                                       output_min_(0);
                    } else {
                        outputs(i, 0) = nn_output;
                    }
                });
        }
    }


    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void fill_input(var_array_t<GRACE_NSPACEDIM>& state ) const {
        int64_t nx,ny,nz ; 
        std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
        int ngz = grace::amr::get_n_ghosts() ; 
        
        int64_t nq = amr::get_local_num_quadrants() ;
        unsigned long index = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq;
        using namespace Kokkos;
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ; 
        Kokkos::parallel_for(GRACE_EXECUTION_TAG("EVOL","get_auxiliaries"), policy 
                , [=,this] GRACE_HOST_DEVICE (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        unsigned long const index = i + (nx + 2*ngz) * (j + (ny + 2*ngz) * (k + (nz + 2*ngz) * q));

        real_t dens = state(i,j,k,DENS_,q);
        real_t tau  = state(i,j,k,TAU_,q);
        real_t sx   = state(i,j,k,SX_,q);
        real_t sy   = state(i,j,k,SY_,q);
        real_t sz   = state(i,j,k,SZ_,q);

        if (has_normalization()) {
            batch_inputs_(index,0) = (dens - input_min_(0)) / (input_max_(0) - input_min_(0));
            batch_inputs_(index,1) = (tau / dens - input_min_(0)) / (input_max_(0) - input_min_(0));
            batch_inputs_(index,2) = (Kokkos::sqrtf(sx*sx + sy*sy + sz*sz) / dens - input_min_(0)) / (input_max_(0) - input_min_(0));
        }
        else {
            batch_inputs_(index,0) = dens;
            batch_inputs_(index,1) = tau / dens;
            batch_inputs_(index,2) = Kokkos::sqrtf(sx*sx + sy*sy + sz*sz) / dens;
        }

    }) ; 
    }

    // Getters (all device-safe)
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    bool is_ready() const { return is_ready_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    int get_input_size() const { return input_size_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    int get_hidden_size() const { return hidden_size_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    int get_output_size() const { return output_size_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    bool has_normalization() const { return has_normalization_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    bool is_physics_guided() const { return is_physics_guided_; }
    
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    real_t get_correction_scale() const { return correction_scale_; }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    Matrix2D get_batch_output() const { return batch_output_; }

    GRACE_HOST_DEVICE
    void reset_batch_allocation(unsigned long batch_size_) {
        Kokkos::realloc(batch_inputs_, batch_size_, input_size_);
        Kokkos::realloc(batch_hidden_pre_activation_, batch_size_, hidden_size_);
        Kokkos::realloc(batch_hidden_post_activation_, batch_size_, hidden_size_);
        Kokkos::realloc(batch_output_pre_activation_, batch_size_, output_size_);
        Kokkos::realloc(batch_output_, batch_size_, output_size_);

        size_t batch_memory = batch_size_ * (input_size_ + 2 * hidden_size_ + 2 * output_size_) * sizeof(real_t);
        GRACE_INFO("ML network batch buffers reallocated:");
        GRACE_INFO("  New batch size: {}", batch_size_);
        GRACE_INFO("  Batch memory: {:.2f} MB", batch_memory / 1024.0 / 1024.0);
    };

private:
    // Network parameters - all Kokkos Views (automatically device-safe)
    Matrix2D weights_input_to_hidden_;
    Vector1D bias_hidden_;
    Matrix2D weights_hidden_to_output_;
    Vector1D bias_output_;
    
    // Normalization parameters
    Vector1D input_min_;
    Vector1D input_max_;
    Vector1D output_min_;
    Vector1D output_max_;

    
    // Network metadata (plain values, device-safe)
    int input_size_ = 0;
    int hidden_size_ = 0;
    int output_size_ = 0;
    bool has_normalization_ = false;
    bool is_physics_guided_ = false;
    real_t correction_scale_ = 0.1;
    bool is_ready_ = false;

    Matrix2D batch_inputs_ ;
    Matrix2D batch_hidden_pre_activation_;   // (batch_size, hidden_size) - nach GEMM, vor tanh
    Matrix2D batch_hidden_post_activation_;  // (batch_size, hidden_size) - nach tanh
    Matrix2D batch_output_pre_activation_;   // (batch_size, output_size) - finales Ergebnis vor denorm
    Matrix2D batch_output_ ;


    /**
     * @brief Physics-based analytical guess for Z (Lorentz factor)
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    real_t analytical_guess(real_t D, real_t q, real_t r) const {
        // Basic kinematic guess: Z ≈ sqrt(1+r^2)
        real_t z_kinematic = Kokkos::sqrt(1.0 + r * r);
        
        // Thermal correction based on internal energy
        real_t thermal_factor = 1.0 + 0.5 * q;
        
        // Relativistic correction for high velocities
        real_t relativistic_correction = 1.0 + 0.1 * r * r / (1.0 + r * r);
        
        // Density correction
        real_t density_correction = 1.0 + 0.1 * Kokkos::log(1.0 + D);
        
        // Combined guess
        real_t z_guess = z_kinematic * thermal_factor * relativistic_correction * density_correction;
        
        // Ensure physical bounds: Z ≥ 1
        return Kokkos::fmax(1.0, z_guess);
    }

    /**
     * @brief Core neural network forward pass
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    real_t neural_network_forward(const real_t* input) const {
        
        // Temporary storage for hidden layer (on stack for small networks)
        real_t hidden[64]; // Adjust size as needed
        
        // Input to hidden layer
        for (int h = 0; h < hidden_size_; ++h) {
            real_t sum = bias_hidden_(h);
            for (int i = 0; i < input_size_; ++i) {
                sum += weights_input_to_hidden_(h, i) * input[i];
            }
            hidden[h] = Kokkos::tanh(sum); // Using tanh activation
        }
        
        // Hidden to output layer
        real_t sum = bias_output_(0);
        for (int h = 0; h < hidden_size_; ++h) {
            sum += weights_hidden_to_output_(0, h) * hidden[h];
        }
        
        return sum; // Linear output activation
    }
};

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML

#endif // GRACE_ML_INFERENCE_NETWORK_HH