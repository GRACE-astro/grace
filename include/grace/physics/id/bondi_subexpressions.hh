
/****************************************************************************/
/*                     Bondi ID helpers, SymPy generated                    */
/****************************************************************************/
#ifndef GRACE_BONDI_ID_SUBEXPR_HH
#define GRACE_BONDI_ID_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
bondi_T__r(
	double M,
	double rc,
	double n,
	double T,
	double r,
	double Tc,
	double uc,
	double * __restrict__ dT
)
{
	double x0 = ((r)*(r)*(r)*(r));
	double x1 = 2*n;
	double x2 = pow(T, x1);
	double x3 = n + 1;
	double x4 = 2*M;
	double x5 = ((uc)*(uc));
	double x6 = x0*x2;
	*dT = (rc*((T*x3 + 1)*(T*x3 + 1))*(pow(Tc, x1)*((rc)*(rc)*(rc)*(rc))*x5 - ((r)*(r)*(r))*x2*x4 + x6) + x6*((Tc*x3 + 1)*(Tc*x3 + 1))*(-rc*(x5 + 1) + x4))/(rc*x0*x2);
}

static void KOKKOS_INLINE_FUNCTION
bondi_ur_rho_p__r(
	double rc,
	double n,
	double K,
	double T,
	double r,
	double Tc,
	double uc,
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
