#ifndef GRACE_C2P_ENTROPY_MHD_HH
#define GRACE_C2P_ENTROPY_MHD_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>

#define SQR(a) (a)*(a)

