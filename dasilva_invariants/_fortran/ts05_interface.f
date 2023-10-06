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
     *        1.,1.,1.,1.,1.,1.,1.,1.,0.)
      end do   
      end     

!     Vectorized TS05 function (scaled version). Accepts arrays of x/y/z
!      coordinates to evaluate model at (with single set of parameters).
!
!     Parameters
!     -----------
!     cf_sf: Chapman-Ferraro Current Scale Factor
!     tail1_sf1: Tail Current #1 Scale Factor
!     tail2_sf1: Tail Current #2 Scale Factor
!     src_sf1: Symmetric Ring Current Scale Factor
!     prc_sf1: Partial Ring Current Scale Factor
!     birk1_sf1: Birkeland Current #1 Scale Factor
!     birk2_sf1: Birkeland Current #2 Scale Factor
!     pen_sf1: Penetrating Field Scale Factor
!     outside_flag: Set to 1 to force NaNs outside the interior
!         magnetosphere (outside the magnetopause). Set to 0 to
!         not do anything special.       
!     ------------------------------------------------------------------------     
      subroutine ts05scalednumpy(PARMOD,PS,X,Y,Z,BX,BY,BZ,n,
     *     cf_sf, tail1_sf, tail2_sf, src_sf, prc_sf,
     *     birk1_sf, birk2_sf, pen_sf, outside_flag)
      real, intent(in) :: PARMOD(10), PS, cf_sf, tail1_sf,
     *     tail2_sf, src_sf, prc_sf, birk1_sf, birk2_sf, pen_sf,
     *     outside_flag
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      
      do i = 1, n
         call T04_s (IOPT,PARMOD,PS,X(i),Y(i),Z(i),BX(i),BY(i),BZ(i),
     *        cf_sf, tail1_sf, tail2_sf, src_sf, prc_sf,
     *        birk1_sf, birk2_sf, pen_sf, outside_flag)
      end do   
      end     
