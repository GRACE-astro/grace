# Test Configuration Files

This directory contains YAML parameter files for RMHD (Relativistic Magnetohydrodynamics) simulations based on Mignone et al. (2021).

## File References

All configurations are based on **Mignone et al. (2021)** - "A comparison of approximate non-linear Riemann solvers for Relativistic MHD" ([arXiv:2111.09369](https://arxiv.org/pdf/2111.09369))

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `rmhd-1.yaml` | Shock Tube 1 | Section 4 / Table 1 |
| `rmhd-2.yaml` | Shock Tube 2 | Section 4 / Table 1 |
| `rmhd-3.yaml` | Shock Tube 3 | Section 4 / Table 1 |
| `rmhd-4.yaml` | Shock Tube 4 | Section 4 / Table 1 |
| `rmhd-cw.yaml` | Circularly polarized wave test | Section 4 / Table 1 |
| `rmhd-rw.yaml` | Rotational waves test | Section 4 / Table 1 |

## Usage when in build

```bash
mpirun -n 1 ./grace --grace-parfile ../config/rmhd-1.yaml

