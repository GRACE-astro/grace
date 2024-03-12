#define magma_DEBUG
#include <thunder/errors/abort.hh>
#include <thunder/errors/error.hh>
#include <thunder/errors/assert.hh>

void test_abort()
{
    ASSERT(0==1, "assertion failed!");
}

void test_error()
{
    // let's send ourselves a SIGSEGV
    int* p;
    *p=1 ;
}

int main()
{
    install_signal_handlers() ;

    test_abort();

    //test_error();
}