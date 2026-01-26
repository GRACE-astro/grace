#ifndef GRACE_AMR_GZ_HELPERS_PR_COEFFS_HH
#define GRACE_AMR_GZ_HELPERS_PR_COEFFS_HH

#include <vector> 

namespace grace { 

namespace detail {

static void fill_fifth_order_restriction_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(20);
    static const double raw_data[20] = {
        -5.0/128.0,
    7.0/32.0,
    -35.0/64.0,
    35.0/32.0,
    35.0/128.0,
    3.0/128.0,
    -5.0/32.0,
    45.0/64.0,
    15.0/32.0,
    -5.0/128.0,
    -5.0/128.0,
    15.0/32.0,
    45.0/64.0,
    -5.0/32.0,
    3.0/128.0,
    35.0/128.0,
    35.0/32.0,
    -35.0/64.0,
    7.0/32.0,
    -5.0/128.0
    };
    coeffs.assign(raw_data, raw_data + 20);
}

static void fill_fifth_order_prolongation_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(10);
    static const double raw_data[10] = {
        -45.0/2048.0,
    105.0/512.0,
    945.0/1024.0,
    -63.0/512.0,
    35.0/2048.0,
    35.0/2048.0,
    -63.0/512.0,
    945.0/1024.0,
    105.0/512.0,
    -45.0/2048.0
    };
    coeffs.assign(raw_data, raw_data + 10);
}

static void fill_fourth_order_restriction_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(12);
    static const double raw_data[12] = {
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
    coeffs.assign(raw_data, raw_data + 12);
}

static void fill_fourth_order_prolongation_coefficients(std::vector<double>& coeffs) {
    coeffs.resize(8);
    static const double raw_data[8] = {
        -5.0/128.0,
    35.0/128.0,
    105.0/128.0,
    -7.0/128.0,
    -7.0/128.0,
    105.0/128.0,
    35.0/128.0,
    -5.0/128.0
    };
    coeffs.assign(raw_data, raw_data + 8);
}

} /*namespace detail */

}

#endif 