/**
 * @file nan_check.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Periodic NaN sanity check on evolved variable arrays.
 *
 * @copyright This file is part of GRACE. See LICENSE.
 */
#ifndef GRACE_SYSTEM_NAN_CHECK_HH
#define GRACE_SYSTEM_NAN_CHECK_HH

#include <grace_config.h>

#include <cstddef>

namespace grace {

/**
 * @brief Scan all evolved variable arrays (cell-centered state, auxiliary,
 *        and staggered face/edge/corner) for NaN values.
 *
 * Performs an MPI Allreduce over ranks so every rank sees the same total.
 *
 * @return Total number of NaN values across all checked arrays and ranks.
 */
std::size_t scan_nans() ;

/**
 * @brief Run a NaN scan if the current iteration warrants one according to
 *        the parameters in <code>nan_check</code>, and take the configured
 *        action ("warn", "terminate", "abort") if any NaNs are found.
 *
 * @param is_initial true when called before the first timestep. Such a call
 *        only runs the scan if <code>nan_check.check_before_first_step</code>
 *        is true. In-loop calls (<code>is_initial=false</code>) only run the
 *        scan if <code>nan_check.check_every</code> is positive and the
 *        current iteration is a multiple of it.
 */
void check_nans_and_act_if_due(bool is_initial) ;

}

#endif /* GRACE_SYSTEM_NAN_CHECK_HH */
