# Major open points:

## Variable storage / handling 
There are several options on how to handle variables within Thunder.
Some of the most reasonable include the following: 

1) Have a gigantic state view that has rank ndim + 2. The ndim ranks 
   encode spatial dimensions in the grid quadrants / octants, the other 
   two ranks are the variable index and the quadrant / octant index. 
   This approach has the benefit that it exposes the maximum possible amount 
   of parallelism. Indeed, variable updates could be as simple as loop over 
   all variables and add arrays together (assuming we have fluxes and rhs already 
   computed, which is unlikely anyway). The other advantage is that the data structures
   in this case are the most simple they could possibly be from the device standpoint:
   just one gigantic array for state (cell-centered, evolved) variables, ndim arrays for 
   face values, ndim for edge values and one for vertex centered variables. The main possible 
   drawback is that the more we make the data monolythic, the more possible penalty we incur 
   in when having to regrid / transfer data, since reallocation of the whole state might be 
   outright impossible. 

2) Have a device "vector" holding the variables. Variables are then rank ndim + 1 views and the 
   variable index becomes the vector index. this means that we can store all evolved variables together
   and separately keep track of their staggering / gz / tl properties. This also has the advantage that 
   any re-sizing / re-allocating of data for regrids or transfers can be done on a per-variable basis.
   this is likely to be slower but will definitely have a smaller memory footprint, which might ultimately 
   be better ( smaller data structures --> smaller memory footprint per octant --> more octants per gpu --> more parallelism // smaller 
   data structures --> less exposed parallelism ).

Overall I prefer approach 1. for now. This seems to me like it's the most straightforward and it does not 
require any fancy data structure which works on device. The design of the variable data struct then would 
look something like this: we will have a singleton class holding a big state view, an auxiliary var view, 
a coordinate view, and some other views for staggered gfs and so on. It will also hold a scratch timelevel 
for t-stepping and a few extra timelevels that can be allocated / de-allocated upon request. The view 
is allocated upon initialization of the singleton and the variable indices are decided at compile time (somehow!)
based on the active physical modules. The variable singleton will also hold a host vector holding some basic 
information about the variables themselvesl, e.g. their name, whether they are staggered, their local index within the
view and so on.
Finally, this class should hold methods that allow the user to obtain a mirror view to the state and / or auxiliary 
variables for I/O purposes. It should also have a way to obtain a subview that only contains the physical / ghost 
points of a given variable.

## Flux storage

The fluxes are obviously computed one direction at a time. It does not (in my opinion), make so much sense to store 
flux arrays at all times. It probably makes more sense to allocate a temporary view containing the fluxes for all 
evolved variables that has ngz = 1 in all directions (assuming we want second order derivatives). Fluxes are not 
staggered gfs in Thunder terminology (makes no sense to alloc/dealloc fluxes within the direction loop, probably very 
slow and only saves on one point in ndim-1 dimensions anyway. )

## State struct 

Might make sense to have a struct which is just a very thin layer around a view that only holds gz / staggering info. 
Maybe or maybe not I'm not sure at all on this point.

AI : easy questions are easy to ask and easy to answer.