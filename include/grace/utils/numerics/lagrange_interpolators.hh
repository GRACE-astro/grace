/**
 * @file lagrange_interpolators.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-21
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */
#ifndef GRACE_UTILS_LAGRANGE_INTERPOLATORS_HH
#define GRACE_UTILS_LAGRANGE_INTERPOLATORS_HH

#include <grace_config.h>
#include <grace/utils/numerics/math.hh>
#include <grace/utils/device/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/numerics/matrix_helpers.tpp>

#include <Kokkos_Core.hpp>

namespace utils {

template< size_t order >
struct corner_staggered_lagrange_interp_t {

    template< size_t idir
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    oned_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) ; 

    template< size_t idir
            , size_t jdir 
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    twod_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) ;

    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) ;

} ; 

/**
 * @brief Edge-centred variables prolongation operators 
 *        
 * \ingroup utils
 * @tparam order Order of the interpolation (since our A-field evolution scheme is 2nd order,
 *                   we only implement 2nd order interpolation now)
 * @tparam edgedir Determines two things here:
 *        1. Which direction is the non-staggered one (e.g. A^z (xc-dx/2,yc-dx/2,zc) with (xc,yc,zc) the
 *           coordinates of the centre)
 *        2. Along which edge or (partially determines the) face for 1d and 2d interplations
 */
template< size_t order, size_t edgedir>
struct edge_staggered_lagrange_interp_t {

    /**
     * @brief 1d Lagrange interpolator for the edge-centred variable
     * \ingroup utils
     * @tparam ichild  Which child edge along the edge (negative direction: 0, positive:1)
     */
    template<size_t ichild
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    oned_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ) ; 

    /**
     * @brief 2d Lagrange interpolator for the edge-centred variable
     * \ingroup utils
     * @tparam ichild  Which child edge among the four across the face (--:0,-+:1,+-:2,++:3)
     * @tparam facedir which face does the interpolation concern, i.e. 
     *                 what is the complementary direction to the one dictated by the vector component (cannot be edgedir!)
     */
    template< size_t ichild
            , size_t facedir 
            , typename view_t >
    requires (edgedir!=facedir)
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    twod_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ) ;

    /**
     * @brief 3d Lagrange interpolator for the edge-centred variable
     * \ingroup utils
     * @tparam ichild  Which child edge among the 2 within the volume (above and below the coarse edge position)
     */
    template<size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ) ;

} ; 

template< size_t order >
struct cell_centered_lagrange_interp_t 
{
    template< size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) ;
} ; 

template< size_t order >
struct generic_lagrange_interpolator_t 
{
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) ;
} ; 

// implementation 

template<> 
struct corner_staggered_lagrange_interp_t<2>
{
    template< size_t idir
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    oned_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        return 0.5*( view(VEC(ic, 
                              jc, 
                              kc)) 
                   + view(VEC(ic+utils::delta(idir,0), 
                              jc+utils::delta(idir,1), 
                              kc+utils::delta(idir,2))) ) ; 
    }

    template< size_t idir
            , size_t jdir 
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    twod_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) {
        using utils::delta; 
        static_assert( not (idir == jdir), "jdir and idir should never coincide in 2D lagrange.") ; 
        return (view(VEC(ic,
                         jc,
                         kc))             
              + view(VEC(ic + delta(0,idir),
                         jc + delta(1,idir),
                         kc + delta(2,idir))) 
              + view(VEC(ic + delta(0,jdir),
                         jc + delta(1,jdir),
                         kc + delta(2,jdir))) 
              + view(VEC(ic + delta(0,idir) + delta(0,jdir),
                         jc + delta(1,idir) + delta(1,jdir),
                         kc + delta(2,idir) + delta(2,jdir))))*0.25 ;
    }

    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) {
        return (view(VEC(ic,jc,kc)) 
              + view(VEC(ic,jc,1 + kc)) 
              + view(VEC(ic,1 + jc,kc)) 
              + view(VEC(ic,1 + jc,1 + kc)) 
              + view(VEC(1 + ic,jc,kc)) 
              + view(VEC(1 + ic,jc,1 + kc)) 
              + view(VEC(1 + ic,1 + jc,kc)) 
              + view(VEC(1 + ic,1 + jc,1 + kc)))*0.125;
    }
};

template<> 
struct corner_staggered_lagrange_interp_t<4>
{
    template< size_t idir
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    oned_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        using utils::delta ;
        return  (-view(VEC(ic - delta(0,idir),jc - delta(1,idir),kc - delta(2,idir))) 
             + 9*(view(VEC(ic,jc,kc)) + view(VEC(ic + delta(0,idir),jc + delta(1,idir),kc + delta(2,idir)))) 
             -    view(VEC(ic + 2*delta(0,idir),jc + 2*delta(1,idir),kc + 2*delta(2,idir))))*0.0625 ; 
    }

    template< size_t idir
            , size_t jdir 
            , typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    twod_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        using utils::delta; 
        static_assert( not (idir == jdir), "jdir and idir should never coincide in 2D lagrange.") ;  
        return (81*view(VEC(ic,jc,kc)) - 9*view(VEC(ic - delta(0,idir),jc - delta(1,idir),kc - delta(2,idir))) + 81*view(VEC(ic + delta(0,idir),jc + delta(1,idir),kc + delta(2,idir))) - 9*view(VEC(ic - delta(0,jdir),jc - delta(1,jdir),kc - delta(2,jdir))) + view(VEC(ic - delta(0,idir) - delta(0,jdir),jc - delta(1,idir) - delta(1,jdir),kc - delta(2,idir) - delta(2,jdir))) - 
     9*view(VEC(ic + delta(0,idir) - delta(0,jdir),jc + delta(1,idir) - delta(1,jdir),kc + delta(2,idir) - delta(2,jdir))) + view(VEC(ic + 2*delta(0,idir) - delta(0,jdir),jc + 2*delta(1,idir) - delta(1,jdir),kc + 2*delta(2,idir) - delta(2,jdir))) + 81*view(VEC(ic + delta(0,jdir),jc + delta(1,jdir),kc + delta(2,jdir))) - 
     9*view(VEC(ic - delta(0,idir) + delta(0,jdir),jc - delta(1,idir) + delta(1,jdir),kc - delta(2,idir) + delta(2,jdir))) + 81*view(VEC(ic + delta(0,idir) + delta(0,jdir),jc + delta(1,idir) + delta(1,jdir),kc + delta(2,idir) + delta(2,jdir))) - 
     9*(view(VEC(ic + 2*delta(0,idir),jc + 2*delta(1,idir),kc + 2*delta(2,idir))) + view(VEC(ic + 2*delta(0,idir) + delta(0,jdir),jc + 2*delta(1,idir) + delta(1,jdir),kc + 2*delta(2,idir) + delta(2,jdir)))) - 9*view(VEC(ic + 2*delta(0,jdir),jc + 2*delta(1,jdir),kc + 2*delta(2,jdir))) + view(VEC(ic - delta(0,idir) + 2*delta(0,jdir),jc - delta(1,idir) + 2*delta(1,jdir),kc - delta(2,idir) + 2*delta(2,jdir))) - 
     9*view(VEC(ic + delta(0,idir) + 2*delta(0,jdir),jc + delta(1,idir) + 2*delta(1,jdir),kc + delta(2,idir) + 2*delta(2,jdir))) + view(VEC(ic + 2*delta(0,idir) + 2*delta(0,jdir),jc + 2*delta(1,idir) + 2*delta(1,jdir),kc + 2*delta(2,idir) + 2*delta(2,jdir))))/256.;
    }

    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        return (-view(VEC(-1 + ic,-1 + jc,-1 + kc)) + 9*view(VEC(-1 + ic,-1 + jc,kc)) + 9*view(VEC(-1 + ic,-1 + jc,1 + kc)) - view(VEC(-1 + ic,-1 + jc,2 + kc)) + 9*view(VEC(-1 + ic,jc,-1 + kc)) - 81*view(VEC(-1 + ic,jc,kc)) - 81*view(VEC(-1 + ic,jc,1 + kc)) + 9*view(VEC(-1 + ic,jc,2 + kc)) + 9*view(VEC(-1 + ic,1 + jc,-1 + kc)) - 81*view(VEC(-1 + ic,1 + jc,kc)) - 81*view(VEC(-1 + ic,1 + jc,1 + kc)) + 
     9*view(VEC(-1 + ic,1 + jc,2 + kc)) - view(VEC(-1 + ic,2 + jc,-1 + kc)) + 9*view(VEC(-1 + ic,2 + jc,kc)) + 9*view(VEC(-1 + ic,2 + jc,1 + kc)) - view(VEC(-1 + ic,2 + jc,2 + kc)) + 9*view(VEC(ic,-1 + jc,-1 + kc)) - 81*view(VEC(ic,-1 + jc,kc)) - 81*view(VEC(ic,-1 + jc,1 + kc)) + 9*view(VEC(ic,-1 + jc,2 + kc)) - 81*view(VEC(ic,jc,-1 + kc)) + 729*view(VEC(ic,jc,kc)) + 729*view(VEC(ic,jc,1 + kc)) - 
     81*view(VEC(ic,jc,2 + kc)) - 81*view(VEC(ic,1 + jc,-1 + kc)) + 729*view(VEC(ic,1 + jc,kc)) + 729*view(VEC(ic,1 + jc,1 + kc)) - 81*view(VEC(ic,1 + jc,2 + kc)) + 9*view(VEC(ic,2 + jc,-1 + kc)) - 81*view(VEC(ic,2 + jc,kc)) - 81*view(VEC(ic,2 + jc,1 + kc)) + 9*view(VEC(ic,2 + jc,2 + kc)) + 9*view(VEC(1 + ic,-1 + jc,-1 + kc)) - 81*view(VEC(1 + ic,-1 + jc,kc)) - 81*view(VEC(1 + ic,-1 + jc,1 + kc)) + 
     9*view(VEC(1 + ic,-1 + jc,2 + kc)) - 81*view(VEC(1 + ic,jc,-1 + kc)) + 729*view(VEC(1 + ic,jc,kc)) + 729*view(VEC(1 + ic,jc,1 + kc)) - 81*view(VEC(1 + ic,jc,2 + kc)) - 81*view(VEC(1 + ic,1 + jc,-1 + kc)) + 729*view(VEC(1 + ic,1 + jc,kc)) + 729*view(VEC(1 + ic,1 + jc,1 + kc)) - 81*view(VEC(1 + ic,1 + jc,2 + kc)) + 9*view(VEC(1 + ic,2 + jc,-1 + kc)) - 81*view(VEC(1 + ic,2 + jc,kc)) - 
     81*view(VEC(1 + ic,2 + jc,1 + kc)) + 9*view(VEC(1 + ic,2 + jc,2 + kc)) - view(VEC(2 + ic,-1 + jc,-1 + kc)) + 9*view(VEC(2 + ic,-1 + jc,kc)) + 9*view(VEC(2 + ic,-1 + jc,1 + kc)) - view(VEC(2 + ic,-1 + jc,2 + kc)) + 9*view(VEC(2 + ic,jc,-1 + kc)) - 81*view(VEC(2 + ic,jc,kc)) - 81*view(VEC(2 + ic,jc,1 + kc)) + 9*view(VEC(2 + ic,jc,2 + kc)) + 9*view(VEC(2 + ic,1 + jc,-1 + kc)) - 81*view(VEC(2 + ic,1 + jc,kc)) - 
     81*view(VEC(2 + ic,1 + jc,1 + kc)) + 9*view(VEC(2 + ic,1 + jc,2 + kc)) - view(VEC(2 + ic,2 + jc,-1 + kc)) + 9*(view(VEC(2 + ic,2 + jc,kc)) + view(VEC(2 + ic,2 + jc,1 + kc))) - view(VEC(2 + ic,2 + jc,2 + kc)))/4096.; 
    }
}; 


template<>
struct cell_centered_lagrange_interp_t<2>
{
    static constexpr double interpolation_coefficients [8][2][2][2] = 
        {
            {//0
                {
                    {125, 75}, 
                    {75, 45}
                }, 
                {
                    {75, 45}, 
                    {45, 27}
                }
            }, 
            {//1
                {
                    {75, 45}, 
                    {45, 27}
                }, 
                {
                    {125, 75}, 
                    {75, 45}
                }
            },
            {//2
                {
                    {75, 45}, 
                    {125, 75}
                }, 
                {
                    {45, 27}, 
                    {75, 45}
                }
            }, 
            {//3
                {
                    {45, 27}, 
                    {75, 45}
                }, 
                {
                    {75, 45}, 
                    {125, 75}
                }
            }, 
            {//4
                {
                    {75, 125}, 
                    {45, 75}
                }, 
                {
                    {45, 75}, 
                    {27, 45}
                }
            }, 
            {//5
                {
                    {45, 75}, 
                    {27, 45}
                }, 
                {
                    {75, 125}, 
                    {45, 75}
                }
            }, 
            {//6
                {
                    {45, 75}, 
                    {75, 125}
                }, 
                {
                    {27, 45}, 
                    {45, 75}
                }
            }, 
            {//7
                {
                    {27, 45}, 
                    {45, 75}
                }, 
                {
                    {45, 75}, 
                    {75, 125}
                }
            }
        };
    template< size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        static constexpr double one_over_denom = 1./512. ; 
        double out = 0 ;
        #pragma unroll 
        for( int i=0; i<2; ++i) for( int j=0; j<2; ++j) for( int k=0; k<2;++k) {
            out += interpolation_coefficients[ichild][i][j][k] * view(VEC(ic-1+i,jc-1+j,kc-1+k)) ; 
        }
        return out * one_over_denom ;  
    }
} ; 
template<>
struct cell_centered_lagrange_interp_t<4>
{

    static constexpr double interpolation_coefficients [8][4][4][4] = 
    {{ // 0
        {
            {-(125), 875, 2625, -(175)}, 
            {875, -(6125), -(18375), 1225}, 
            {2625, -(18375), -(55125), 3675}, 
            {-(175), 1225, 3675, -(245)}
        }, 
        {
            {875, -(6125), -(18375), 1225}, 
            {-(6125), 42875, 128625, -(8575)}, 
            {-(18375), 128625, 385875, -(25725)}, 
            {1225, -(8575), -(25725), 1715}
        }, 
        {
            {2625, -(18375), -(55125), 3675}, 
            {-(18375), 128625, 385875, -(25725)}, 
            {-(55125), 385875, 1157625, -(77175)}, 
            {3675, -(25725), -(77175), 5145}
        }, 
        {
            {-(175), 1225, 3675, -(245)},  
            {1225, -(8575), -(25725), 1715}, 
            {3675, -(25725), -(77175), 5145}, 
            {-(245), 1715, 5145, -(343)}
        }    
    }, 
    { // 1
        {
            {125, -(875), -(2625), 175}, 
            {-(875), 6125, 18375, -(1225)}, 
            {-(2625), 18375, 55125, -(3675)}, 
            {175, -(1225), -(3675), 245}
        }, 
        {
            {-(675), 4725, 14175, -(945)}, 
            {4725, -(33075), -(99225), 6615}, 
            {14175, -(99225), -(297675), 19845}, 
            {-(945), 6615, 19845, -(1323)}
        }, 
        {
            {3375, -(23625), -(70875), 4725}, 
            {-(23625), 165375, 496125, -(33075)}, 
            {-(70875), 496125, 1488375, -(99225)}, 
            {4725, -(33075), -(99225), 6615}
        }, 
        {
            {375, -(2625), -(7875), 525}, 
            {-(2625), 18375, 55125, -(3675)}, 
            {-(7875), 55125, 165375, -(11025)}, 
            {525, -(3675), -(11025), 735}
        }
    },
    { // 2
        {
            {125, -(875), -(2625), 175}, 
            {-(675), 4725, 14175, -(945)}, 
            {3375, -(23625), -(70875), 4725}, 
            {375, -(2625), -(7875), 525}
        }, 
        {
            {-(875), 6125, 18375, -(1225)}, 
            {4725, -(33075), -(99225), 6615}, 
            {-(23625), 165375, 496125, -(33075)}, 
            {-(2625), 18375, 55125, -(3675)}
        }, 
        {
            {-(2625), 18375, 55125, -(3675)}, 
            {14175, -(99225), -(297675), 19845}, 
            {-(70875), 496125, 1488375, -(99225)}, 
            {-(7875), 55125, 165375, -(11025)}
        }, 
        {
            {175, -(1225), -(3675), 245}, 
            {-(945), 6615, 19845, -(1323)}, 
            {4725, -(33075), -(99225), 6615}, 
            {525, -(3675), -(11025), 735}
        }
    },
    { // 3 
        {
            {-(125), 875, 2625, -(175)}, 
            {675, -(4725), -(14175), 945}, 
            {-(3375), 23625, 70875, -(4725)}, 
            {-(375), 2625, 7875, -(525)}
        }, 
        {
            {675, -(4725), -(14175), 945}, 
            {-(3645), 25515, 76545, -(5103)}, 
            {18225, -(127575), -(382725), 25515}, 
            {2025, -(14175), -(42525), 2835}
        }, 
        {
            {-(3375), 23625, 70875, -(4725)}, 
            {18225, -(127575), -(382725), 25515}, 
            {-(91125), 637875, 1913625, -(127575)}, 
            {-(10125), 70875, 212625, -(14175)}
        }, 
        {
            {-(375), 2625, 7875, -(525)}, 
            {2025, -(14175), -(42525), 2835}, 
            {-(10125), 70875, 212625, -(14175)}, 
            {-(1125), 7875, 23625, -(1575)}
        }
    },
    { // 4
        {
            {125, -(675), 3375, 375}, 
            {-(875), 4725, -(23625), -(2625)}, 
            {-(2625), 14175, -(70875), -(7875)}, 
            {175, -(945), 4725, 525}
        }, 
        {
            {-(875), 4725, -(23625), -(2625)}, 
            {6125, -(33075), 165375, 18375}, 
            {18375, -(99225), 496125, 55125}, 
            {-(1225), 6615, -(33075), -(3675)}
        }, 
        {
            {-(2625), 14175, -(70875), -(7875)}, 
            {18375, -(99225), 496125, 55125}, 
            {55125, -(297675), 1488375, 165375}, 
            {-(3675), 19845, -(99225), -(11025)}
        }, 
        {
            {175, -(945), 4725, 525}, 
            {-(1225), 6615, -(33075), -(3675)}, 
            {-(3675), 19845, -(99225), -(11025)}, 
            {245, -(1323), 6615, 735}
        }
    },
    { // 5 
        {
            {-(125), 675, -(3375), -(375)}, 
            {875, -(4725), 23625, 2625},
            {2625, -(14175), 70875, 7875}, 
            {-(175), 945, -(4725), -(525)}
        }, 
        {
            {675, -(3645), 18225, 2025}, 
            {-(4725), 25515, -(127575), -(14175)}, 
            {-(14175), 76545, -(382725), -(42525)}, 
            {945, -(5103), 25515, 2835}
        }, 
        {
            {-(3375), 18225, -(91125), -(10125)}, 
            {23625, -(127575), 637875, 70875}, 
            {70875, -(382725), 1913625, 212625}, 
            {-(4725), 25515, -(127575), -(14175)}
        }, 
        {
            {-(375), 2025, -(10125), -(1125)}, 
            {2625, -(14175), 70875, 7875}, 
            {7875, -(42525), 212625, 23625}, 
            {-(525), 2835, -(14175), -(1575)}
        }
    }, 
    { // 6
        {
            {-(125), 675, -(3375), -(375)}, 
            {675, -(3645), 18225, 2025}, 
            {-(3375), 18225, -(91125), -(10125)}, 
            {-(375), 2025, -(10125), -(1125)}
        }, 
        {
            {875, -(4725), 23625, 2625}, 
            {-(4725), 25515, -(127575), -(14175)}, 
            {23625, -(127575), 637875, 70875}, 
            {2625, -(14175), 70875, 7875}
        }, 
        {
            {2625, -(14175), 70875, 7875}, 
            {-(14175), 76545, -(382725), -(42525)}, 
            {70875, -(382725), 1913625, 212625}, 
            {7875, -(42525), 212625, 23625}
        }, 
        {
            {-(175), 945, -(4725), -(525)}, 
            {945, -(5103), 25515, 2835}, 
            {-(4725), 25515, -(127575), -(14175)}, 
            {-(525), 2835, -(14175), -(1575)}
        }
    }, 
    { // 7 
        {
            {125, -(675), 3375, 375}, 
            {-(675), 3645, -(18225), -(2025)}, 
            {3375, -(18225), 91125, 10125}, 
            {375, -(2025), 10125, 1125}
        }, 
        {
            {-(675), 3645, -(18225), -(2025)}, 
            {3645, -(19683), 98415, 10935}, 
            {-(18225), 98415, -(492075), -(54675)}, 
            {-(2025), 10935, -(54675), -(6075)}
        }, 
        {
            {3375, -(18225), 91125, 10125}, 
            {-(18225), 98415, -(492075), -(54675)}, 
            {91125, -(492075), 2460375, 273375}, 
            {10125, -(54675), 273375, 30375}
        }, 
        {
            {375, -(2025), 10125, 1125}, 
            {-(2025), 10935, -(54675), -(6075)}, 
            {10125, -(54675), 273375, 30375}, 
            {1125, -(6075), 30375, 3375}
        }
    }
    } ; 

    template< size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ic, int jc, int kc)
    ) 
    {
        static constexpr double one_over_denom = 1./2097152. ; 
        double out = 0 ;
        #pragma unroll 
        for( int i=0; i<4; ++i) for( int j=0; j<4; ++j) for( int k=0; k<4;++k) {
            out += interpolation_coefficients[ichild][i][j][k] * view(VEC(ic-2+i,jc-2+j,kc-2+k)) ; 
        }
        return out * one_over_denom ;  
    }
} ; 



template <int edgedir>
consteval std::tuple<int, int> get_complementary_dirs() {
    constexpr std::array<std::tuple<int, int>, 3> complementary_dirs = {{
        {1, 2}, // if edgedir == 0
        {0, 2}, // if edgedir == 1
        {0, 1}  // if edgedir == 2
    }};
    return complementary_dirs[edgedir];
}

// implementation:
// template paramter size_t ichild 
// inquires for the child 
template <size_t edgedir>
struct edge_staggered_lagrange_interp_t<2,edgedir>{
    static constexpr int idir = std::get<0>(get_complementary_dirs<edgedir>());
    static constexpr int jdir = std::get<1>(get_complementary_dirs<edgedir>());

    // one coarse edge contributes to 2 fine edges along the edge 
    // size_t ichild = 0 is below, 1 is the fine edge above the coarse edge 
    template< size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    oned_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ){
        if constexpr(ichild==0){
            return (3.0 * view(VEC(ie, 
                            je,
                            ke)) 
                + view(VEC(ie-utils::delta(edgedir,0), 
                            je-utils::delta(edgedir,1), 
                            ke-utils::delta(edgedir,2))) ) ;
        }
        else if constexpr(ichild==1){
            return (3.0 * view(VEC(ie, 
                            je, 
                            ke)) 
                + view(VEC(ie+utils::delta(edgedir,0), 
                            je+utils::delta(edgedir,1), 
                            ke+utils::delta(edgedir,2))) ) ;
        } 
        else{
            static_assert(false);
            }
    } 

    // size_t ichild = 0 is below, 1 is the fine edge above the coarse edge 
    // one coarse edge contributes to 4 fine edges across the face
    template< size_t ichild,size_t facedir, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    twod_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ){
        static_assert( not (edgedir == facedir), "edgedir and facedir should never coincide in 2D lagrange.") ;  
        using utils::delta;

             if constexpr(ichild==0){
               return (3*view(VEC(ie,je,ke)) +\
                         view(VEC(ie - delta(0,edgedir),je - delta(1,edgedir),ke - delta(2,edgedir))) +\
                       3*view(VEC(ie + delta(0,facedir),je + delta(1,facedir),ke + delta(2,facedir))) + \
                         view(VEC(ie - delta(0,edgedir) + delta(0,facedir),je - delta(1,edgedir) + delta(1,facedir),ke - delta(2,edgedir) + delta(2,facedir))))/8.;
        }
        else if constexpr(ichild==1){
               return (3*view(VEC(ie,je,ke)) +\
                         view(VEC(ie + delta(0,edgedir),je + delta(1,edgedir),ke + delta(2,edgedir))) + \
                       3*view(VEC(ie + delta(0,facedir),je + delta(1,facedir),ke + delta(2,facedir))) + \
                         view(VEC(ie + delta(0,edgedir) + delta(0,facedir),je + delta(1,edgedir) + delta(1,facedir),ke + delta(2,edgedir) + delta(2,facedir))))/8.;
        }
        else static_assert(false);
    }
    // each edge in coarse view (ie,je,ke)
    // contributes to 2 child edges (in fine view) that are located in the general volume (i.e. not along the edge or on a face)

    template< size_t ichild, typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    threed_interp(
        view_t& view, VEC(int ie, int je, int ke)
    ){
               using utils::delta;

              if constexpr(ichild==0){
              return (3*view(VEC(ie,je,ke)) + view(VEC(1 + ie,1 + je,1 + ke)) + view(VEC(ie + delta(0,edgedir),je + delta(1,edgedir),ke + delta(2,edgedir))) + 
                        3*view(VEC(ie + delta(0,idir),je + delta(1,idir),ke + delta(2,idir))) + 
                        view(VEC(ie + delta(0,edgedir) + delta(0,idir),je + delta(1,edgedir) + delta(1,idir),ke + delta(2,edgedir) + delta(2,idir))) + 
                        3*view(VEC(ie + delta(0,jdir),je + delta(1,jdir),ke + delta(2,jdir))) + 
                        view(VEC(ie + delta(0,edgedir) + delta(0,jdir),je + delta(1,edgedir) + delta(1,jdir),ke + delta(2,edgedir) + delta(2,jdir))) + 
                        3*view(VEC(ie + delta(0,idir) + delta(0,jdir),je + delta(1,idir) + delta(1,jdir),ke + delta(2,idir) + delta(2,jdir))))/16.;
        }else if constexpr(ichild==1){
              return (3*view(VEC(ie,je,ke)) + view(VEC(ie + delta(0,edgedir),je + delta(1,edgedir),ke + delta(2,edgedir))) + 
                        3*view(VEC(ie + delta(0,idir),je + delta(1,idir),ke + delta(2,idir))) + 
                        view(VEC(ie + delta(0,edgedir) + delta(0,idir),je + delta(1,edgedir) + delta(1,idir),ke + delta(2,edgedir) + delta(2,idir))) + 
                        3*view(VEC(ie + delta(0,jdir),je + delta(1,jdir),ke + delta(2,jdir))) + 
                        view(VEC(ie + delta(0,edgedir) + delta(0,jdir),je + delta(1,edgedir) + delta(1,jdir),ke + delta(2,edgedir) + delta(2,jdir))) + 
                        3*view(VEC(ie + delta(0,idir) + delta(0,jdir),je + delta(1,idir) + delta(1,jdir),ke + delta(2,idir) + delta(2,jdir))) + 
                        view(VEC(ie + delta(0,edgedir) + delta(0,idir) + delta(0,jdir),je + delta(1,edgedir) + delta(1,idir) + delta(1,jdir),
                        ke + delta(2,edgedir) + delta(2,idir) + delta(2,jdir))))/16.;
        }
    
    }

} ; 




}
#endif /* GRACE_UTILS_LAGRANGE_INTERPOLATORS_HH */