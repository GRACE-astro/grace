
/****************************************************************************/
/*                     Bondi ID helpers, SymPy generated                    */
/****************************************************************************/
#ifndef GRACE_BONDI_ID_SUBEXPR_HH
#define GRACE_BONDI_ID_SUBEXPR_HH

//#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
bondi_T__r(
	double M,
	double T,
	double Tc,
	double n,
	double r,
	double rc,
	double uc,
	double* __restrict__ f,
	double* __restrict__ df
)
{
	double x0 = n + 1;
	double x1 = ((uc)*(uc));
	double x2 = 2*M;
	double x3 = T*x0 + 1;
	double x4 = 2*n;
	double x5 = pow(T, x4);
	double x6 = ((r)*(r)*(r)*(r));
	double x7 = 1/(x6);
	double x8 = pow(Tc, x4)*((rc)*(rc)*(rc)*(rc))*x1;
	*f = ((x3)*(x3))*(1 + x7*x8/x5 - x2/r) - ((Tc*x0 + 1)*(Tc*x0 + 1))*(x1 + 1 - x2/rc);
	*df = 2*pow(T, -4*n - 1)*x3*x7*(pow(T, x4 + 1)*x0*(-((r)*(r)*(r))*x2*x5 + x5*x6 + x8) - n*x3*x5*x8);
	
}

static void KOKKOS_INLINE_FUNCTION
bondi_ur_rho_p__r(
	double K,
	double T,
	double Tc,
	double n,
	double r,
	double rc,
	double uc,
	double* __restrict__ ur,
	double* __restrict__ rho,
	double* __restrict__ p
)
{
	double x0 = pow(T/K, n);
	*ur = pow(T, -n)*pow(Tc, n)*((rc)*(rc))*uc/((r)*(r));
	*rho = x0;
	*p = T*x0;
	
}

static void KOKKOS_INLINE_FUNCTION
bondi_uc_Tc(
	double M,
	double n,
	double rc,
	double* __restrict__ uc,
	double* __restrict__ Tc
)
{
	double x0 = M/rc;
	double x1 = (1.0/2.0)*x0;
	*uc = -1.0/2.0*M_SQRT2*sqrt(x0);
	*Tc = n*x1/((n + 1)*(-x1*(n + 3) + 1));
	
}

#endif 
