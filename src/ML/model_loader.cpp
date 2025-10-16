#include "grace/ML/model_loader.hh"

#include "grace/system/print.hh"
#include <hdf5.h>
#include <vector>

namespace grace {
namespace ml {

bool ModelLoader::load_model_from_hdf5(
    const std::string& model_file,
    NetworkConfig& config,
    HostMatrix2D& h_weights_ih,
    HostVector1D& h_bias_h,
    HostMatrix2D& h_weights_ho,
    HostVector1D& h_bias_o) 
{
    try {
        hid_t file_id = H5Fopen(model_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            spdlog::error("Failed to open HDF5 file: {}", model_file);
            return false;
        }

        if (!read_metadata_attributes(file_id, config)) {
            H5Fclose(file_id);
            return false;
        }

        h_weights_ih = HostMatrix2D("host_weights_ih", config.hidden_size, config.input_size);
        h_bias_h = HostVector1D("host_bias_h", config.hidden_size);
        h_weights_ho = HostMatrix2D("host_weights_ho", config.output_size, config.hidden_size);
        h_bias_o = HostVector1D("host_bias_o", config.output_size);

        if (!read_dataset_2d(file_id, "weights_input_to_hidden", h_weights_ih) ||
            !read_dataset_1d(file_id, "bias_hidden", h_bias_h) ||
            !read_dataset_2d(file_id, "weights_hidden_to_output", h_weights_ho) ||
            !read_dataset_1d(file_id, "bias_output", h_bias_o)) {
            H5Fclose(file_id);
            return false;
        }

        if (H5Lexists(file_id, "input_min", H5P_DEFAULT) && 
            H5Lexists(file_id, "input_max", H5P_DEFAULT)) {
            config.input_min = HostVector1D("host_input_min", config.input_size);
            config.input_max = HostVector1D("host_input_max", config.input_size);
            if (read_dataset_1d(file_id, "input_min", config.input_min) &&
                read_dataset_1d(file_id, "input_max", config.input_max)) {
                config.has_normalization = true;
            }
        }

        if (H5Lexists(file_id, "output_min", H5P_DEFAULT) && 
            H5Lexists(file_id, "output_max", H5P_DEFAULT)) {
            config.output_min = HostVector1D("host_output_min", config.output_size);
            config.output_max = HostVector1D("host_output_max", config.output_size);
            read_dataset_1d(file_id, "output_min", config.output_min);
            read_dataset_1d(file_id, "output_max", config.output_max);
        }

        H5Fclose(file_id);

        GRACE_INFO("Successfully loaded {} model from {}", config.model_type, model_file);
        GRACE_INFO("Architecture: {}->{}->{}", config.input_size, config.hidden_size, config.output_size);
        if (config.is_physics_guided) {
            GRACE_INFO("Correction scale: {}", config.correction_scale);
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception while loading model from {}: {}", model_file, e.what());
        return false;
    }
}

// --- helpers (no duplication of structs/types!) ---

bool ModelLoader::read_metadata_attributes(hid_t file_id, NetworkConfig& config) {
    if (!read_int_attribute(file_id, "input_size", config.input_size) ||
        !read_int_attribute(file_id, "hidden_size", config.hidden_size) ||
        !read_int_attribute(file_id, "output_size", config.output_size)) {
        return false;
    }
    if (read_double_attribute(file_id, "correction_scale", config.correction_scale)) {
        config.is_physics_guided = true;
    }
    read_string_attribute(file_id, "activation_function", config.activation_function);
    read_string_attribute(file_id, "model_type", config.model_type);
    if (config.model_type == "physics_guided") {
        config.is_physics_guided = true;
    }
    return true;
}

bool ModelLoader::read_int_attribute(hid_t file_id, const std::string& name, int& value) {
    if (!H5Aexists(file_id, name.c_str())) return false;
    hid_t attr_id = H5Aopen(file_id, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) return false;
    herr_t status = H5Aread(attr_id, H5T_NATIVE_INT, &value);
    H5Aclose(attr_id);
    return status >= 0;
}

bool ModelLoader::read_double_attribute(hid_t file_id, const std::string& name, double& value) {
    if (!H5Aexists(file_id, name.c_str())) return false;
    hid_t attr_id = H5Aopen(file_id, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) return false;
    herr_t status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr_id);
    return status >= 0;
}

bool ModelLoader::read_string_attribute(hid_t file_id, const std::string& name, std::string& value) {
    if (!H5Aexists(file_id, name.c_str())) return false;
    hid_t attr_id = H5Aopen(file_id, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) return false;
    hid_t type = H5Aget_type(attr_id);
    size_t size = H5Tget_size(type);
    std::vector<char> buffer(size + 1, '\0');
    herr_t status = H5Aread(attr_id, type, buffer.data());
    if (status >= 0) value = buffer.data();
    H5Tclose(type);
    H5Aclose(attr_id);
    return status >= 0;
}

bool ModelLoader::read_dataset_2d(hid_t file_id, const std::string& name, HostMatrix2D& mat) {
    hid_t dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) return false;
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());
    H5Dclose(dataset_id);
    return status >= 0;
}

bool ModelLoader::read_dataset_1d(hid_t file_id, const std::string& name, HostVector1D& vec) {
    hid_t dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) return false;
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    H5Dclose(dataset_id);
    return status >= 0;
}

} // namespace ml
} // namespace grace
