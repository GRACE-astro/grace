#include <iostream>

#define GRACE_NSPACEDIM 3

int main() 
{
    int const ngz{2} ;
    int const nx{6}, ny{6}, nz{6} ; 

    auto const index_mapping = [&] ( int const iig, 
                                    int const jjg, 
                                    int const kk, 
                                    int const ea, 
                                    int const eb, 
                                    int ijk[GRACE_NSPACEDIM], 
                                    int lmn[GRACE_NSPACEDIM] ) 
    {
        static constexpr const int ALONG_EDGE = -1 ;
        static constexpr const int NEGATIVE_EDGE = 0 ;
        static constexpr const int POSITIVE_EDGE = 1 ;
        static const int edge_directions[3][12] = {
            {ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, 
            NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE,
            NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE},  // x directions

            {NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, 
            ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE,
            NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE}, // y directions

            {NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
            NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
            ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE} // z directions
        };   
        
        // Extract directional values for edges ea and eb
        int x_ea = edge_directions[0][ea];
        int y_ea = edge_directions[1][ea];
        int z_ea = edge_directions[2][ea];

        // Map indices for ijk based on edge ea
        ijk[0] = (x_ea == ALONG_EDGE) ? (kk+ngz) : (x_ea == NEGATIVE_EDGE ? iig : (nx + ngz + iig));
        ijk[1] = (y_ea == ALONG_EDGE) 
                    ? (kk+ngz) 
                    : (x_ea == ALONG_EDGE 
                        ? (y_ea == NEGATIVE_EDGE ? iig : (ny + ngz + iig)) 
                        : (y_ea == NEGATIVE_EDGE ? jjg : (ny + ngz + jjg)));
        ijk[2] = (z_ea == ALONG_EDGE) ? (kk+ngz) : (z_ea == NEGATIVE_EDGE ? jjg : (nz + ngz + jjg));

        int x_eb = edge_directions[0][eb];
        int y_eb = edge_directions[1][eb];
        int z_eb = edge_directions[2][eb];

        // Map indices for lmn based on edge eb
        lmn[0] = (x_eb == ALONG_EDGE) ? ((2*kk)%nx+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (nx - ngz + 2*iig));
        lmn[1] = (y_eb == ALONG_EDGE) 
                    ? ((2*kk)%ny+ngz)
                    : (x_eb == ALONG_EDGE 
                        ? (y_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (ny - ngz + 2*iig)) 
                        : (y_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (ny - ngz + 2*jjg)));
        lmn[2] = (z_eb == ALONG_EDGE) ? ((2*kk)%nz+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (nz - ngz + 2*jjg));
    } ;

    int index_pairs[6][2] = {
        {0,3},{1,2},{4,7},{5,6},{8,11},{9,10}
    } ; 

    for ( int ip=0; ip<6; ++ip ) {
        int ijk[3],lmn[3] ; 
        index_mapping(0,0,0,index_pairs[ip][0],index_pairs[ip][1],ijk,lmn) ; 
        std::cout << "Edge A " << index_pairs[ip][0] << std::endl ; 
        std::cout << "Edge B " << index_pairs[ip][1] << std::endl ; 
        std::cout << "ijk_c " << std::endl ; 
        std::cout << ijk[0] << ", " << ijk[1] << ", " << ijk[2] << std::endl ;
        std::cout << "ijk_f " << std::endl ;
        std::cout << lmn[0] << ", " << lmn[1] << ", " << lmn[2] << std::endl ;
        std::cout << std::endl; 
    }
}