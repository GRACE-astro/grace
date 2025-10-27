register_grace_object(grace_c2p c2p.cpp)
#target_compile_options(grace_c2p PRIVATE
#    -fno-fast-math
#    -fno-unsafe-math-optimizations
#)
register_grace_object(grace_eos eos_storage.cpp)
