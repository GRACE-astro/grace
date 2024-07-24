#include <utility>
#include <tuple>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cassert> 
#include <omp.h>

#define GRACE_HOST
#define GRACE_ALWAYS_INLINE inline __attribute__((always_inline))

template<typename... Args>
struct are_pairs_of_size_t {
    static constexpr bool value = std::conjunction<std::is_same<Args, std::pair<std::size_t, std::size_t>>...>::value;
};

template<typename... Args>
constexpr bool are_pairs_of_size_t_v = are_pairs_of_size_t<Args...>::value ;

template< typename Tuple 
        , std::size_t ... Is >
auto GRACE_ALWAYS_INLINE GRACE_HOST 
index_unpacker(std::size_t index, Tuple const& ranges, std::index_sequence<Is...> ) {
    std::tuple<decltype(Is)...> result;

    auto unpack = [&](auto&... indices) {
        size_t sizes[] = { (std::get<Is>(ranges).second - std::get<Is>(ranges).first)... };
        (..., (indices = (index % sizes[Is] + std::get<Is>(ranges).first), index /= sizes[Is]));
    };

    std::apply(unpack, result);
    return result;
}

template<typename Tuple, std::size_t... Is>
std::size_t get_cumulative_range(const Tuple& ranges, std::index_sequence<Is...>) {
    return (... * (std::get<Is>(ranges).second - std::get<Is>(ranges).first));
}

/**
 * @brief Implementation of host loop with closure as body
 * \ingroup utils 
 * \cond grace_detail
 * @tparam omp_parallel Whether the loop should be OpenMP parallelized.
 */
template< bool omp_parallel >
struct host_loop_impl_t {
    template< typename Ft
            , typename ... Idxt > 
    void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt && ... ranges ) ; 
} ; 

template<>
struct host_loop_impl_t<false> {

    template< typename Ft
            , typename ... Idxt >
    static void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt && ... args ) {
        static_assert( are_pairs_of_size_t_v<Idxt...>
                     , "Loop index ranges must be provided as "
                       "a parameter pack that can be interpreted "
                       "as a list of std::pair<std::size_t,std::size_t>.") ; 
        static constexpr const std::size_t ndim = sizeof...(Idxt) ; 
        auto const ranges = std::make_tuple(args...) ; 
        const size_t ncumulative = get_cumulative_range(ranges,std::index_sequence_for<Idxt...>{}) ;
        for (std::size_t i = 0UL; i < ncumulative; i+=1UL) {
            auto indices = index_unpacker(i, ranges, std::index_sequence_for<Idxt...>{}) ; 
            std::apply(_func, indices) ; 
        }
    }

} ; 

template<>
struct host_loop_impl_t<true> {

    template< typename Ft
            , typename ... Idxt >
    static void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt && ... args ) {
        static_assert( are_pairs_of_size_t_v<Idxt...>
                     , "Loop index ranges must be provided as "
                       "a parameter pack that can be interpreted "
                       "as a list of std::pair<std::size_t,std::size_t>.") ; 
        static constexpr const std::size_t ndim = sizeof...(Idxt) ; 
        auto const ranges = std::make_tuple(args...) ; 
        const size_t ncumulative = get_cumulative_range(ranges,std::index_sequence_for<Idxt...>{}) ;
        #pragma omp parallel for 
        for (std::size_t i = 0UL; i < ncumulative; i+=1UL) {
            auto indices = unpack_indices(i, ranges, std::index_sequence_for<Idxt...>{}) ; 
            std::apply(_func, indices) ; 
        }
    }

} ;



template< bool omp_parallel
        , typename Ft 
        , typename ... Idxt > 
void GRACE_ALWAYS_INLINE GRACE_HOST 
host_ndloop(Ft&& _func, Idxt&& ... args ) {
    host_loop_impl_t<omp_parallel>::template loop<Ft>(
          std::forward<Ft>(_func)
        , std::forward<Idxt>(args)...
    ) ; 
}

int main() {

    size_t * x = (size_t*) malloc( sizeof(size_t) * 100 * 100 ) ; 

    auto const loop_body = [=] ( size_t i, size_t j) {
        size_t idx = j + 100* i ; 
        x[idx] = idx ; 
    } ; 
    std::pair<std::size_t,std::size_t> range {0,100} ; 
    host_ndloop<false>(loop_body, std::move(range), std::pair<std::size_t,std::size_t>{0,100}) ; 

    for( int i=0 ; i < 100*100; ++i) {
        assert(x[i] == i) ; 
    }

    free(x) ; 

}