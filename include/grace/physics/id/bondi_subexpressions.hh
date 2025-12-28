
/****************************************************************************/
/*                     Bondi ID helpers, SymPy generated                    */
/****************************************************************************/
#ifndef GRACE_BONDI_ID_SUBEXPR_HH
#define GRACE_BONDI_ID_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
bondi_T__r(
	double n,
	double uc,
	double r,
	double Tc,
	double T,
	double M,
	double rc,
	double * __restrict__ dT,
	double * __restrict__ dT_dr
)
{
	double x0 = n + 1;
	double x1 = 2*M;
	double x2 = ((uc)*(uc));
	double x3 = ((r)*(r)*(r)*(r));
	double x4 = 2*n;
	double x5 = pow(T, x4);
	double x6 = x3*x5;
	double x7 = x6*((Tc*x0 + 1)*(Tc*x0 + 1))*(-rc*(x2 + 1) + x1);
	double x8 = T*x0;
	double x9 = x8 + 1;
	double x10 = ((x9)*(x9));
	double x11 = ((r)*(r)*(r))*x5;
	double x12 = rc*(pow(Tc, x4)*((rc)*(rc)*(rc)*(rc))*x2 - x1*x11 + x6);
	double x13 = x10*x12 + x7;
	double x14 = 1/(rc*x3);
	*dT = x13*x14/x5;
	*dT_dr = 2*pow(T, -x4 - 1)*x14*(-n*rc*x10*x11*(-r + x1) - n*x13 + n*x7 + x12*x8*x9);
}

static void KOKKOS_INLINE_FUNCTION
bondi_ur_rho_p__r(
	double Tc,
	double T,
	double n,
	double rc,
	double uc,
	double r,
	double K,
	double * __restrict__ ur,
	double * __restrict__ rho,
	double * __restrict__ press
)
{
	double x0 = pow(T/K, n);
	*ur = pow(T, -n)*pow(Tc, n)*((rc)*(rc))*uc/((r)*(r));
	*rho = x0;
	*press = T*x0;
}

static void KOKKOS_INLINE_FUNCTION
bondi_uc_Tc(
	double M,
	double rc,
	double n,
	double * __restrict__ ur_c,
	double * __restrict__ T_c
)
{
	double x0 = M/rc;
	double x1 = (1.0/2.0)*x0;
	*ur_c = -1.0/2.0*M_SQRT2*sqrt(x0);
	*T_c = -n*x1/((n + 1)*(x1*(n + 3) - 1));
}

#endif 
