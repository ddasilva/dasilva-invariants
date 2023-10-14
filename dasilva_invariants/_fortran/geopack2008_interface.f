!     This code written by Daniel da Silva. It contains functions that
!     are exposed to python using f2py. Each function here must be
!     listed in the funcs argument to _get_extension() in setup.py
!     ================================================================

!     Wrapper for the recalc_08() subroutine
!       time = (year, doy, hour, minute month)
!       v = (vx, vy, vz)
!     --------------------------------------- ---
      subroutine recalc(time, v)
      integer, intent(in) :: time(5)
      real, intent(in) :: v(3)
      
      call recalc_08 (time(1), time(2), time(3), time(4),
     * time(5), v(1), v(2), v(3))

      end subroutine

!     Wrapper to get the dipole field in SM coordinates
!     -------------------------------------------------
      subroutine dipnumpy(X,Y,Z,BX,BY,BZ,n)
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      real :: xgsw, ygsw, zgsw, bxgsw, bygsw, bzgsw

      do i = 1, n
         call SMGSW_08(X(i), Y(i), Z(i), xgsw, ygsw, zgsw, 1)
         call DIP_08(xgsw, ygsw, zgsw, bxgsw, bygsw, bzgsw)
         call SMGSW_08(BX(i), BY(i), BZ(i), bxgsw, bygsw, bzgsw, -1)         
      end do
      end subroutine

!     Wrapper around the IGRF field code in SM coordinates
!     -------------------------------------------------      
      subroutine igrfnumpy(X,Y,Z,BX,BY,BZ,n)
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      real :: xgsw, ygsw, zgsw, bxgsw, bygsw, bzgsw

      do i = 1, n
         call SMGSW_08(X(i), Y(i), Z(i), xgsw, ygsw, zgsw, 1)
         call IGRF_GSW_08(xgsw, ygsw, zgsw, bxgsw, bygsw, bzgsw)
         call SMGSW_08(BX(i), BY(i), BZ(i), bxgsw, bygsw, bzgsw, -1)         
      end do
      
      end subroutine
