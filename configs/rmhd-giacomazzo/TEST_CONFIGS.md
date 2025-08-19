# Test Configuration Files

This directory contains YAML parameter files for RMHD (Relativistic Magnetohydrodynamics) simulations based on Giacomazzo & Rezzolla (2007).

## File References

All configurations are based on **Giacomazzo & Rezzolla (2006)** - "The exact solution of the Riemann problem in relativistic magnetohydrodynamics" ([arXiv:gr-qc/0507102](https://arxiv.org/pdf/gr-qc/0507102))

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `rmhd-bal-1.yaml` | Balsara Test 1 | Section 6.2, Test 1 |
| `rmhd-bal-2.yaml` | Balsara Test 2 | Section 6.2, Test 2 |
| `rmhd-bal-3.yaml` | Balsara Test 3 | Section 6.2, Test 3 |
| `rmhd-bal-4.yaml` | Balsara Test 4 | Section 6.2, Test 4 |
| `rmhd-bal-5.yaml` | Balsara Test 5 | Section 6.2, Test 5 |
| `rmhd-generic-afven.yaml` | Generic Alfvén wave test | Section 6.2 |
| `rmhd-ko-sh-1.yaml` | Komissarov shock tube test 1 | Section 6.2, Test 1 |
| `rmhd-ko-sh-coll.yaml` | Komissarov: Collision Test | Section 4.4 |

## Usage when in build

```bash
mpirun -n 1 ./grace --grace-parfile ../config/rmhd-bal-1.yaml