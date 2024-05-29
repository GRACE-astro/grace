#ifndef AEC986E5_311F_4FDF_9E46_3ABDF632C6FD
#define AEC986E5_311F_4FDF_9E46_3ABDF632C6FD

#include <sc.h> 

#ifdef GRACE_3D
#include <p4est_to_p8est.h>
#include <p8est.h>
#include <p8est_connectivity.h> 
#include <p8est_communication.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#include <p8est_balance.h>
#include <p8est_ghost.h>
#include <p8est_iterate.h>
#include <p8est_search.h>
#else
#include <p4est.h>
#include <p4est_connectivity.h> 
#include <p4est_communication.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#include <p4est_base.h>
#include <p4est_balance.h>
#include <p4est_ghost.h>
#include <p4est_iterate.h>
#include <p4est_search.h>
#endif 

#endif /* AEC986E5_311F_4FDF_9E46_3ABDF632C6FD */
