/*----------------
 destributed in Code: flags
#ifdef GRACE_EANBLE_MHD_Apot
#endif
#define VARIABLES_LIST_MHD_BASE 
//MHD_TODO
-------------------*/


//------- piece from auxiliaries.cpp -----------------------------
void compute_auxiliary_quantities(){
    //.......
    #ifdef GRACE_ENABLE_MHD_Apot
    //#--------------------------------------
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        //MHD_TODO : something like edge staggered policy?
        //MHD_TODO : other zones than nx+1+ngz,ny+1+ngz,nz+1+ngz
        edge_staggered_policy({VEC(ngz,ngz,ngz),0},{VEC(nx+1+ngz,ny+1+ngz,nz+1+ngz),nq}) ; 
        parallel_for(GRACE_EXECUTION_TAG("EVOL","get_auxiliaries"), corner_staggered_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
           .....
    }
    //#-------------------------------------
    //MHD_TODO: this is for now here: not complete & likely wrong: check edge vs face
    template <typename eos_t>
    void compute_magnetic_field(
    var_array_t<GRACE_NSPACEDIM>& state,
    var_array_t<GRACE_NSPACEDIM>& aux,
    staggered_variable_arrays_t& sstate) {

    using namespace grace;
    using namespace Kokkos;

    int nx, ny, nz;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents();
    int ngz = amr::get_n_ghosts();
    int nq = amr::get_local_num_quadrants();

    parallel_for(
        GRACE_EXECUTION_TAG("AUX", "compute_B"),
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>, default_execution_space>(
            {VEC(0, 0, 0), 0}, {VEC(nx + 2 * ngz, ny + 2 * ngz, nz + 2 * ngz), nq}),
        KOKKOS_LAMBDA(VEC(int i, int j, int k), int q) {
            double dAx_dy = (sstate(VEC(i, j + 1, k), APOTX, q) - sstate(VEC(i, j - 1, k), APOTX, q)) / (2.0 * dx);
            double dAx_dz = (sstate(VEC(i, j, k + 1), APOTX, q) - sstate(VEC(i, j, k - 1), APOTX, q)) / (2.0 * dx);

            double dAy_dx = (sstate(VEC(i + 1, j, k), APOTY, q) - sstate(VEC(i - 1, j, k), APOTY, q)) / (2.0 * dx);
            double dAy_dz = (sstate(VEC(i, j, k + 1), APOTY, q) - sstate(VEC(i, j, k - 1), APOTY, q)) / (2.0 * dx);

            double dAz_dx = (sstate(VEC(i + 1, j, k), APOTZ, q) - sstate(VEC(i - 1, j, k), APOTZ, q)) / (2.0 * dx);
            double dAz_dy = (sstate(VEC(i, j + 1, k), APOTZ, q) - sstate(VEC(i, j - 1, k), APOTZ, q)) / (2.0 * dx);

            aux(VEC(i, j, k), BMAGX, q) = dAy_dz - dAz_dy;
            aux(VEC(i, j, k), BMAGY, q) = dAz_dx - dAx_dz;
            aux(VEC(i, j, k), BMAGZ, q) = dAx_dy - dAy_dx;
        });
    }
    #endif
 //....
    }
//------- initial data routine from grmhd.cpp -----------------------
//==============================================================
#ifdef GRACE_ENABLE_MHD_Apot
template <typename eos_t
        , typename id_t 
        , typename ... arg_t > 
void set_initial_magnetic_potential(arg_t ... kernel_args) {
    using namespace grace ;
    using namespace Kokkos ; 
    auto& estate = grace::variable_list::get().getstaggeredstate().edge_staggered_fields ;

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;

    
    auto const Id_from_params = get_param<double>("grmhd","Id_from_params") ; 

    //-------------- option 1: id from parameters ----------------
    // if: Bi,Apoti params:
    if(Id_from_params){
       auto const BX1_id = get_param<double>("grmhd","Bx1_initial") ; 
       auto const BY1_id = get_param<double>("grmhd","By1_initial") ; 
       auto const BZ1_id = get_param<double>("grmhd","Bz1_initial") ; 
       
       auto const BX2_id = get_param<double>("grmhd","Bx2_initial") ; 
       auto const BY2_id = get_param<double>("grmhd","By2_initial") ; 
       auto const BZ2_id = get_param<double>("grmhd","Bz2_initial") ; 
       
       auto const APOTX1_id = get_param<double>("grmhd","Ax1_initial") ; 
       auto const APOTY1_id = get_param<double>("grmhd","Ay1_initial") ; 
       auto const APOTZ1_id = get_param<double>("grmhd","Az1_initial") ; 
    }
    //-------------------------------------------------------------

    //------ for aux: option 1: fill up h_mirrow and deepcopy --------
    if(Id_from_params){
       auto& coord_system = grace::coordinate_system::get() ; 
       auto h_state_mirror = Kokkos::create_mirror_view(aux) ; 
   
       int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
       #pragma omp parallel for 
       for( int64_t icell=0; icell<ncells; ++icell) {
           size_t const i = icell%(nx+2*ngz); 
           size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
           #ifdef GRACE_3D 
           size_t const k = 
               (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
           size_t const q = 
               (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
           #else 
           size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
           #endif 
           /* Physical coordinates of cell center */
           auto pcoords = coord_system.get_physical_coordinates(
               {VEC(i,j,k)},
               q,
               true
           ) ; 
           // for id after some radius
           double const r = 
	       Kokkos::sqrt( EXPR( pcoords[0]*pcoords[0], + pcoords[1]*pcoords[1], + pcoords[2] * pcoords[2])) ;
   
           h_state_mirror(VEC(i,j,k),BX,q) = 0. ;
           h_state_mirror(VEC(i,j,k),BY,q) = 0. ;
           h_state_mirror(VEC(i,j,k),BZ,q) = 0. ;
           // distriubtion of B up to user
           if ( pcoords[0] <= 0 ) {
               h_state_mirror(VEC(i,j,k),BX,q) = BX1_id ;
               h_state_mirror(VEC(i,j,k),BY,q) = BY1_id ;
               h_state_mirror(VEC(i,j,k),BZ,q) = BZ1_id ; 
           } else {
               h_state_mirror(VEC(i,j,k),BX,q) = BX2_id ;
               h_state_mirror(VEC(i,j,k),BY,q) = BY2_id ;
               h_state_mirror(VEC(i,j,k),BZ,q) = BZ2_id ; 
           }
       }
       Kokkos::deep_copy(aux,h_state_mirror) ;
    }
   // -----------------------------------------------------------

    // ---- option 2: as id_t ----------------
    if (!Id_from_params) {
       auto const& _eos = eos::get().get_eos<eos_t>() ; 
       id_t id_kernel{ _eos, pcoords, kernel_args... } ; 
    }
    parallel_for(
        GRACE_EXECUTION_TAG("ID", "set_Ai")
        , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
        , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
        {
            // option 2: id_t --------------
            if (!Id_from_params) {
               auto const id = id_kernel(VEC(i,j,k), q) ; 
               
               estate(VEC(i, j, k), APOTX_, q) = id.Apotx; /* Initial condition for Ax */
               estate(VEC(i, j, k), APOTY_, q) = id.Apoty;/* Initial condition for Ay */
               estate(VEC(i, j, k), APOTZ_, q) = id.Apotz;/* Initial condition for Az */
  
               aux(VEC(i, j, k), BX_, q) = id.Bx;             
               aux(VEC(i, j, k), BY_, q) = id.By;            
               aux(VEC(i, j, k), BZ_, q) = id.Bz; 
            }//-------------------------
            else{
               estate(VEC(i, j, k), APOTX_, q) = APOTX1_id; /* Initial condition for Ax */
               estate(VEC(i, j, k), APOTY_, q) = APOTY1_id;/* Initial condition for Ay */
               estate(VEC(i, j, k), APOTZ_, q) = APOTY1_id;/* Initial condition for Az */
   
               // this aux fill up would overwrite the mirrored aux initialization
               //aux(VEC(i, j, k), BX_, q) = BX1_id;             
               //aux(VEC(i, j, k), BY_, q) = BY1_id;            
               //aux(VEC(i, j, k), BZ_, q) = BZ1_id; 
            }
        });
}
#endif
//===============================================


//--------- piece from grmhd_magn.hh : Apot-flux---------
void compute_mhd_fluxes(){
      #ifdef GRACE_EANBLE_MHD_Apot
                //MHD_TODO: computation of induction equation rhs
                // should be done separately in evolve using reconstructed vels
        /***********************************************************************/ 
        /*                           Get A_x flux                              */
        /***********************************************************************/
        /* F^d_{A_x} = v^z * \tilde{B}^y - v^y * \tilde{B}^z                   */
        /***********************************************************************/
        /* prim[B] has to be \tilde{B} = \sqrtg * B at the cell center*/ 
        /* prim[V] has to be the reconstructed velocity !! */
        fl = primL[VZL] * primL[BMAGY] - primL[VYL] * primL[BMAGZ]; //=epsBx_L
        fr = primR[VZL] * primR[BMAGY] - primR[VYL] * primR[BMAGZ]; //epsBx_R
        // or
        //double const epsBx_L = primL[VZL] * primL[BMAGY] - primL[VYL] * primL[BMAGZ]; 
        //double const epsBx_R = primR[VZL] * primR[BMAGY] - primR[VYL] * primR[BMAGZ]; 

        //U_state(A_x), MHD_TODO: in this routine only prims exist, no cons!
        // worng for now:
        double const epsBx_L = primL[BMAGX]; //!! but it should be = Apotx !
        double const epsBx_R = primR[BMAGX];
      
        //MHD_TODO define APOTXL
        f[APOTXL] = solver(fl,fr,epsBx_L,epsBx_R,cmin,cmax) ; 

        /***********************************************************************/ 
        /*                           Get A_y flux                              */
        /***********************************************************************/
        /* F^d_{A_y} = v^x * \tilde{B}^z - v^z * \tilde{B}^x                   */
        /***********************************************************************/
        /* prim[B] has to be \tilde{B} = \sqrtg * B at the cell center*/ 
        /* prim[V] has to be the reconstructed velocity !!! */
        fl = primL[VXL] * primL[BMAGZ] - primL[VZL] * primL[BMAGX]; 
        fr = primR[VXL] * primR[BMAGZ] - primR[VZL] * primR[BMAGX]; 
        
        //U_state(A_y), MHD_TODO: in this routine only prims exist, no cons!
        // worng for now:
        double const epsBy_L = primL[BMAGY]; 
        double const epsBy_R = primR[BMAGY]; 
    
        //MHD_TODO define APOTXL
        f[APOTYL] = solver(fl,fr,epsBx_L,epsBx_R,cmin,cmax) ; 
        /***********************************************************************/ 
        /*                           Get A_z flux                              */
        /***********************************************************************/
        /* F^d_{A_z} = v^y * \tilde{B}^x - v^x * \tilde{B}^y                   */
        /***********************************************************************/
        /* prim[B] has to be \tilde{B} = \sqrtg * B at the cell center*/ 
        /* prim[V] has to be the reconstructed velocity !!! */
        fl = primL[VYL] * primL[BMAGX] - primL[VXL] * primL[BMAGY]; 
        fr = primR[VYL] * primR[BMAGX] - primR[VXL] * primR[BMAGY]; 
        
        //U_state(A_x), MHD_TODO: in this routine only prims exist, no cons!
        // worng for now:
        double const epsBy_L = primL[BMAGZ]; 
        double const epsBy_R = primR[BMAGZ]; 
    
        //MHD_TODO define APOTXL
        f[APOTZL] = solver(fl,fr,epsBx_L,epsBx_R,cmin,cmax) ; 
        /***********************************************************************/
        #endif
}

//--------- piece from grmhd.hh : compute smallb,b2 -------------------------
    /***********************************************************************/
    /**
     * @brief Compute b2 and smallb
     * 
     * @param smallb Comoving magnetic field
     * @param b2  Square of comoving magnetic field.
     * @param u0  zeroth component of 4-velocity.
     * @param prims Primitive variables.
     * @param metric Metric tensor.
        // need metric_face here 
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_smallb( double& smallb,  double const& b2, double const& u0
               , grace::grmhd_prims_array_t const& prims 
               , grace::metric_array_t const& metric) const
    {
        /* Get 4 metric   */
        auto const gmunu = metric.gmunu() ;                                                                                */
        //auto const gamma = metric.gamma() ;
        // following gmunu of /utils/numerics/metric_utils.hh
        auto gtt_ = 0;
        auto gtx_ = 1;
        auto gty_ = 2;
        auto gtz_ = 3;
        auto gxx_ = 4;
        auto gxy_ = 5;
        auto gxz_ = 6;
        auto gyy_ = 7;
        auto gyz_ = 8;
        auto gzz_ = 9;

        // Notes: here gmunu has to be conformally flat?

        /* Calculate v^i + beta^i */
        double const vx_plus_betax = prims[VXL] + metric.beta(0);
        double const vy_plus_betay = prims[VYL] + metric.beta(1);
        double const vz_plus_betaz = prims[VZL] + metric.beta(2);

        /* Calculate u_i / (u_0 \psi^4)*/
        double const u_x_over_u0_psi4r =  gmunu[gxx_]*vx_plus_betax 
                       + gmunu[gxy_]*vy_plus_betay + gmunu[gxz_]*vz_plus_betaz;
        double const u_y_over_u0_psi4r =  gmunu[gxy_]*vx_plus_betax 
                       + gmunu[gyy_]*vy_plus_betay + gmunu[gyz_]*vz_plus_betaz;  
        double const u_z_over_u0_psi4r =  gmunu[gxz_]*vx_plus_betax 
                       + gmunu[gyz_]*vy_plus_betay + gmunu[gzz_]*vz_plus_betaz;

        // MHD-TODO get B-field
        double Bx_center = prims[BMAGX];
        double By_center = prims[BMAGY];
        double Bz_center = prims[BMAGZ];

        /* Calculate \alpha*\sqrt{4\pi} = u_i B^i 
                      = u_x/(u_0 \psi^4) B^x + u_y/(u_0 \psi^4) B^y + u_z/(u_0 \psi^4) B^z*/
        double const ulow_i_Bup_i = u_x_over_u0_psi4r * Bx_center 
                       + u_y_over_u0_psi4r * By_center + u_z_over_u0_psi4r * Bz_center;

        /* Calculate smallb: the comoving magnetic field components b^{\mu}*/
        //MHD_TODO: get pi?
        smallb[1] = (Bx_center *1./u0 + prims[VXL] * ulow_i_Bup_i ) * 1./metric.alpha() /sqrt(4. * pi);
        smallb[2] = (By_center *1./u0 + prims[VYL] * ulow_i_Bup_i ) * 1./metric.alpha() /sqrt(4. * pi);
        smallb[3] = (Bz_center *1./u0 + prims[VZL] * ulow_i_Bup_i ) * 1./metric.alpha() /sqrt(4. * pi);
        smallb[0] = ulow_i_Bup_i * 1./metric.alpha() /sqrt(4. * pi);

        /* Calculate the square b2 of the comoving magnetic field */
        b2 = gmunu[gtt_] * math::int_pow<2>(smallb[0]) + gmunu[gxx_] * math::int_pow<2>(smallb[1]) 
           + gmunu[gyy_] * math::int_pow<2>(smallb[2]) + gmunu[gzz_] * math::int_pow<2>(smallb[3]) 
           + 2. * ( gmunu[gtx_] * smallb[0] * smallb[1] + gmunu[gty_] * smallb[0] * smallb[2]
                  + gmunu[gtz_] * smallb[0] * smallb[3] + gmunu[gxy_] * smallb[1] * smallb[2]
                  + gmunu[gxz_] * smallb[1] * smallb[3] + gmunu[gyz_] * smallb[2] * smallb[3]);

    }
    /***********************************************************************/