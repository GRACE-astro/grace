#ifndef GRACE_ML_INFERENCE_MODEL_LOADER_HH
#define GRACE_ML_INFERENCE_MODEL_LOADER_HH

#include <Kokkos_Core.hpp>
#include <grace_config.h>

#ifdef GRACE_ENABLE_ML

#include "grace/ML/network_config.hh"
#include <string>
#include <spdlog/spdlog.h>

namespace grace {

// Define memory space here (same as in device_network.hh)
#if defined(GRACE_ENABLE_CUDA)
using default_space_ai = Kokkos::CudaSpace;  
#elif defined(GRACE_ENABLE_HIP)
using default_space_ai = Kokkos::HIPSpace;
#elif defined(GRACE_ENABLE_OMP) || defined(GRACE_ENABLE_SERIAL)
using default_space_ai = Kokkos::HostSpace;
#endif   

namespace ml {
class ModelLoader {
public:
    static bool load_model_from_hdf5(const std::string& model_file,
                                    NetworkConfig& config,
                                    HostMatrix2D& h_weights_ih,
                                    HostVector1D& h_bias_h,
                                    HostMatrix2D& h_weights_ho,
                                    HostVector1D& h_bias_o) ;

private:
    static bool read_metadata_attributes(long file_id, NetworkConfig& config) ;
    static bool read_int_attribute(long file_id, const std::string& attr_name, int& value) ;
    static bool read_double_attribute(long file_id, const std::string& attr_name, double& value) ;
    static bool read_string_attribute(long file_id, const std::string& attr_name, std::string& value) ;
    static bool read_dataset_2d(long file_id, const std::string& dataset_name, HostMatrix2D& matrix) ;
    static bool read_dataset_1d(long file_id, const std::string& dataset_name, HostVector1D& vector) ;
};

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML

#endif // GRACE_ML_INFERENCE_MODEL_LOADER_HH