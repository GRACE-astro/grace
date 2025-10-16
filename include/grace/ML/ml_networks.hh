/**
 * @file ml_network_list.hh
 * @brief ML network storage and management
 */

#ifndef GRACE_ML_INFERENCE_ML_NETWORKS
#define GRACE_ML_INFERENCE_ML_NETWORKS

#include <grace_config.h>

#ifdef GRACE_ENABLE_ML

#include <Kokkos_Core.hpp>
#include <grace/utils/inline.h>
#include <grace/utils/singleton_holder.hh>
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh>

#include "grace/ML/device_network.hh"

namespace grace {
namespace ml {

/**
 * @brief Implementation of ML network list
 * 
 * Manages all ML inference networks used in GRACE.
 * Wrapped with singleton_holder to ensure proper lifetime.
 */
class ml_network_list_impl_t {
public:    
    /**
     * @brief Get C2P network (device view)
     * 
     * @return Device view containing the network
     */
    GRACE_ALWAYS_INLINE 
    Kokkos::View<MLInferenceNetwork*>& get_c2p_device_view() { 
        return _c2p_network; 
    }
    
    /**
     * @brief Get C2P network (const device view)
     * 
     * @return Const device view containing the network
     */
    GRACE_ALWAYS_INLINE 
    const Kokkos::View<MLInferenceNetwork*>& get_c2p_device_view() const { 
        return _c2p_network; 
    }

    /**
     * @brief Check if C2P network is initialized
     * @return true if network is ready for use
     */
    GRACE_ALWAYS_INLINE
    bool is_c2p_initialized() const {
        return _c2p_network.size() > 0;
    }

private:
    /**
     * @brief Construct the ml_network_list_impl_t
     * 
     * Reads configuration and initializes all ML networks
     */
    ml_network_list_impl_t();
    
    /**
     * @brief Destroy the ml_network_list_impl_t
     */
    ~ml_network_list_impl_t() = default;
    
    //******** Member variables ***************************************************************************
    
    Kokkos::View<MLInferenceNetwork*> _c2p_network;  //!< Device view of C2P network
    
    // Add more networks here as needed:
    // MLInferenceNetwork _p2c_network_host;
    // Kokkos::View<MLInferenceNetwork*> _p2c_network_device;
    
    //******** Singleton infrastructure ***********************************************************
    
    friend class utils::singleton_holder<ml_network_list_impl_t, memory::default_create>;
    friend class memory::new_delete_creator<ml_network_list_impl_t, memory::new_delete_allocator>;
    
    friend class utils::singleton_holder<ml_network_list_impl_t, memory::default_create>;
    friend class memory::new_delete_creator<ml_network_list_impl_t, memory::new_delete_allocator>;
    static constexpr size_t longevity = GRACE_ML_NETWORKS;

};

/**
 * @brief Proxy holding all ML networks within GRACE.
 * 
 * Only a unique instance exists at runtime.
 * Access via ml_network_list::get()
 */
using ml_network_list = utils::singleton_holder<ml_network_list_impl_t>;

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML

#endif // GRACE_ML_INFERENCE_ML_NETWORKS