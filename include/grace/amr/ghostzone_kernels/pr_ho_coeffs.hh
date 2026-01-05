#ifndef GRACE_AMR_GZ_HELPERS_PR_COEFFS_HH
#define GRACE_AMR_GZ_HELPERS_PR_COEFFS_HH

#include <vector> 

namespace grace { 

namespace detail {

static void fill_fourth_order_restriction_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(168);
    static const double raw_data[168] = {
        1.0/16.0,
    -5.0/16.0,
    15.0/16.0,
    5.0/16.0,
    -1.0/16.0,
    9.0/16.0,
    9.0/16.0,
    -1.0/16.0,
    5.0/16.0,
    15.0/16.0,
    -5.0/16.0,
    1.0/16.0
    };
    coeffs.assign(raw_data, raw_data + 168);
}

static void fill_fourth_order_prolongation_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(124);
    static const double raw_data[124] = {
        -5.0/128.0,
    35.0/128.0,
    105.0/128.0,
    -7.0/128.0,
    -7.0/128.0,
    105.0/128.0,
    35.0/128.0,
    -5.0/128.0
    };
    coeffs.assign(raw_data, raw_data + 124);
}

} /*namespace detail */

}

#endif 