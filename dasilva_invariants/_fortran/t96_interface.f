!     This code written by Daniel da Silva. It contains functions that
!     are exposed to python using f2py. Each function here must be
!     listed in the funcs argument to _get_extension() in setup.py
!     ================================================================
      
      subroutine t96numpy(PARMOD,PS,X,Y,Z, BX,BY,BZ,n)
      real, intent(in) :: PARMOD(10), PS
      real, intent(in) :: X(n), Y(n), Z(n)     
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      
      do i = 1, n
         call T96_01(IOPT,PARMOD,PS,X(i),Y(i),Z(i),BX(i),BY(i),BZ(i))
      end do   
      end
