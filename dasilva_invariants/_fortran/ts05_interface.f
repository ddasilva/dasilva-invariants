!     This code written by Daniel da Silva. It contains functions that
!     are exposed to python using f2py. Each function here must be
!     listed in the funcs argument to _get_extension() in setup.py
!     ================================================================

!     Vectorized TS05 function. Accepts arrays of x/y/z coordinates to 
!     evaluate model at (with single set of parameters).
!     ------------------------------------------------------------------------
      subroutine ts05numpy(PARMOD,PS,X,Y,Z,BX,BY,BZ,n)
      real, intent(in) :: PARMOD(10), PS
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      
      do i = 1, n
         call T04_s (IOPT,PARMOD,PS,X(i),Y(i),Z(i),BX(i),BY(i),BZ(i),
     *        1.,1.,1.,1.,1.,1.,1.,1.)
      end do   
      end     

!     Vectorized TS05 function. Accepts arrays of x/y/z coordinates to 
!     evaluate model at (with single set of parameters).
!     ------------------------------------------------------------------------     
      subroutine ts05scalednumpy(PARMOD,PS,X,Y,Z,BX,BY,BZ,n,
     *     cf_sf, tail1_sf, tail2_sf, src_sf, prc_sf,
     *     birk1_sf, birk2_sf, pen_sf)
      real, intent(in) :: PARMOD(10), PS, cf_sf, tail1_sf,
     *     tail2_sf, src_sf, prc_sf, birk1_sf, birk2_sf, pen_sf      
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      
      do i = 1, n
         call T04_s (IOPT,PARMOD,PS,X(i),Y(i),Z(i),BX(i),BY(i),BZ(i),
     *        cf_sf, tail1_sf, tail2_sf, src_sf, prc_sf,
     *        birk1_sf, birk2_sf, pen_sf)
      end do   
      end     
