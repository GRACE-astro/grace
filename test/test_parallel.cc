#include <catch2/catch_test_macros.hpp>
#include <grace/parallel/mpi_wrappers.hh>
#include <iostream>
TEST_CASE("MPI wrappers", "[mpi_wrappers]")
{   
    using namespace parallel ;
    auto myP = mpi_comm_rank() ; 
    auto nP  = mpi_comm_size() ;
    
    SECTION("size_rank") {
        REQUIRE(myP < nP ) ; 
    }
    
    SECTION("gather") {
        int buf = myP ;
        int size_send = 1; 
        int * recv_buffer = nullptr ;
        if ( myP == 0 ) 
            recv_buffer = (int*) malloc( nP * sizeof(int) ) ;
        mpi_gather(&buf, size_send, recv_buffer, 
            size_send ) ;
        if( myP==0 ){
            for( int i=0; i<nP*size_send; ++i)
                REQUIRE( recv_buffer[i] == i) ;
            free(recv_buffer);
        }
    }
    
    SECTION("gatherv")
    {   
        int send_size = 10 ;
        int stride = 12 ;
        int * send_buffer = (int*) malloc(send_size*sizeof(int)) ;
        int * send_sizes = (int*) malloc(nP*sizeof(int)) ;
        int * displs = (int*) malloc(nP*sizeof(int)) ;
        for(int i=0; i<send_size; ++i) {
            send_buffer[i] = myP ;
        } 
        for(int i=0; i<nP; ++i){
            send_sizes[i] = 10 ;
            displs[i] = i * stride ;
        }
        int * recv_buffer = nullptr;
        if (myP == 0){
            recv_buffer = (int*) malloc(nP * stride * sizeof(int)) ;   
        }
        mpi_gatherv(send_buffer, send_size, recv_buffer, send_sizes, displs) ;
        if(myP==0){
            for( int i=0; i<nP; ++i)
            {
                for(int j=0; j<send_sizes[i]; ++j)
                    REQUIRE( recv_buffer[ displs[i] + j ] == i) ;
            }
            free(recv_buffer);
        }
        free(send_buffer); free(send_sizes); free(displs);
    }
    
    SECTION("allgather")
    {
        int buf = myP ;
        int size_send = 1; 
        int * recv_buffer = (int*) malloc( nP * sizeof(int) ) ;
        mpi_allgather(&buf, size_send, recv_buffer, 
            size_send ) ;
        for( int i=0; i<nP*size_send; ++i)
            REQUIRE( recv_buffer[i] == i) ;
        free(recv_buffer);
    }
    
    SECTION("allgatherv")
    {   
        int send_size = 10 ;
        int stride = 12 ;
        int * send_buffer = (int*) malloc(send_size*sizeof(int)) ;
        int * send_sizes = (int*) malloc(nP*sizeof(int)) ;
        int * displs = (int*) malloc(nP*sizeof(int)) ;
        for(int i=0; i<send_size; ++i) {
            send_buffer[i] = myP ;
        } 
        for(int i=0; i<nP; ++i){
            send_sizes[i] = 10 ;
            displs[i] = i * stride ;
        }
        int * recv_buffer = (int*) malloc(nP * stride * sizeof(int)) ;           
        mpi_allgatherv(send_buffer, send_size, recv_buffer, send_sizes, displs) ;
        for( int i=0; i<nP; ++i)
        {
            for(int j=0; j<send_sizes[i]; ++j)
                REQUIRE( recv_buffer[ displs[i] + j ] == i) ;
        }
        free(recv_buffer);
        free(send_buffer); free(send_sizes); free(displs);
    }
    SECTION("alltoall")
    {   
        int send_size = 1 ;
        int recv_size = 1 ;
        int * send_buffer = (int*) malloc( nP*sizeof(int)) ;
        for(int i=0; i<nP; ++i) {
            send_buffer[i] = myP + i * 100 ;
        } 
        int * recv_buffer = (int*) malloc(nP * sizeof(int)) ;           
        mpi_alltoall(send_buffer, send_size, recv_buffer, recv_size) ;
        for( int i=0; i<nP; ++i)
        {   
            REQUIRE( recv_buffer[ i ] == myP*100 + i ) ;
        }
        free(recv_buffer);
        free(send_buffer);
    }
    
    SECTION("bcast")
    {   
        int buf ;
        if(myP == 0) {
            buf = 42;
        }
        mpi_bcast(&buf,1,0);
        REQUIRE(buf == 42) ;
    }
    
    SECTION("reduce") {
        int buf = 1 ;
        int count = 1; 
        int sum ;
        mpi_reduce(&buf, &sum, count, sc_MPI_SUM) ;
        if( myP==0 ){
            REQUIRE(sum==nP);
        }
    }

    SECTION("allreduce") {
        int buf = 1 ;
        int count = 1; 
        int sum ;
        mpi_allreduce(&buf, &sum, count, sc_MPI_SUM) ;
        REQUIRE(sum==nP);
    }
    
}