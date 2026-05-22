#!/usr/bin/env bash
# Sweep the partner-symmetry check across the evolved Z4 variables to
# localize which one breaks equivariance first.  Prints a single-line
# summary per variable: first-rel (at the earliest t>0), last-rel,
# and worst-rel across the run.
#
# Usage:
#   ./sweep_z4_symmetry.sh <descriptor.xmf> [transform]
#
# Defaults: transform = pirot_z.  Override check_symmetry.py location with
# CHECK_SYMMETRY=/path/to/check_symmetry.py.
#
# Author: carlo.musolino@aei.mpg.de

set -euo pipefail

DESC="${1:?usage: $0 <descriptor.xmf> [transform]}"
TRANS="${2:-pirot_z}"
CHECK="${CHECK_SYMMETRY:-$HOME/check_symmetry.py}"

# Discover the variable names actually present in this descriptor and
# print them up front, so missing-name issues are obvious.
echo "Variables in $DESC:"
grep -oE 'Name="[^"]+"' "$DESC" | sort -u | sed 's/Name=/  /; s/"//g'
echo

# Evolved Z4 variables exposed in the slice output.  Names taken from the
# defaults in examples/symmetry_audit/tools/plot_asymmetry_map.py.  Edit / extend as needed for
# variables that exist in your specific run (e.g. gtdd / Atdd as symmetric
# tensors require component-level checks not currently supported by
# check_symmetry.py's --kind=scalar/polar/axial).
PAIRS=(
    "alp:scalar"
    "conf_fact:scalar"
    "z4c_theta:scalar"
    "z4c_Khat:scalar"
    "z4c_Gamma:polar"
    "z4c_Bdriver:polar"
    "beta:polar"
)

printf "%-18s %-8s %-12s %-12s %-12s\n" "variable" "kind" "first-rel" "last-rel" "worst-rel"
printf "%-18s %-8s %-12s %-12s %-12s\n" "--------" "----" "---------" "--------" "---------"

for pair in "${PAIRS[@]}"; do
    var="${pair%%:*}"
    kind="${pair##*:}"
    # Don't treat nonzero exit as an error: check_symmetry.py exits
    # nonzero when *any* time-slice fails the tol check, which is exactly
    # the case we want to characterise.  Parse the output unconditionally
    # and only flag a real error if no rel lines came back at all.
    out=$(python "$CHECK" --var "$var" --kind "$kind" \
                          --time all --transform "$TRANS" "$DESC" 2>&1 || true)
    rels=$(printf '%s\n' "$out" | grep -oE 'max\|Δ\|=[0-9.eE+-]+[[:space:]]+scale=[0-9.eE+-]+[[:space:]]+rel=[0-9.eE+-]+' \
            | grep -oE 'rel=[0-9.eE+-]+' | sed 's/^rel=//')
    if [ -z "$rels" ]; then
        first_err=$(printf '%s\n' "$out" | head -3 | tr '\n' ' ')
        printf "%-18s %-8s ERROR: %s\n" "$var" "$kind" "$first_err"
        continue
    fi
    # skip the t=0 zero on first-rel
    first=$(printf '%s\n' "$rels" | awk 'NR>1 {print; exit}')
    last=$(printf '%s\n'  "$rels" | tail -1)
    worst=$(printf '%s\n' "$rels" | sort -g | tail -1)
    printf "%-18s %-8s %-12s %-12s %-12s\n" \
        "$var" "$kind" "${first:-—}" "${last:-—}" "${worst:-—}"
done
