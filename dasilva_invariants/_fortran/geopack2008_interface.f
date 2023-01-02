!     This code written by Daniel da Silva. It contains functions that
!     are exposed to python using f2py. Each function here must be
!     listed in the funcs argument to _get_extension() in setup.py
!     ================================================================

!     Wrapper for the recalc_08() subroutine
!        time = (year, doy, hour, minute month)
!     --------------------------------------- 
      subroutine recalc(time, vx, vy, vz)
      integer, intent(in) :: time(5)
      real, intent(in) :: vx, vy, vz
      
      call recalc_08 (time(1), time(2), time(3), time(4),
     * time(5), vx, vy, vz)

      end subroutine

!     Forces the Dipole Tilt to a certain value
!     -----------------------------------------
      subroutine force_dipole_tilt(dipole_tilt)
      real, intent(in) :: dipole_tilt
      common /GEOPACK1/ PSI,SPS,CPS

      PSI = dipole_tilt
      CPS = COS(dipole_tilt)
      SPS = SIN(dipole_tilt)
      end subroutine

!     Wrapper to get the dipole field
!     --------------------------------
      subroutine dipnumpy(X,Y,Z,BX,BY,BZ,n)
      real, intent(in) :: X(n), Y(n), Z(n)
      integer, intent(in) :: n
      real, intent(out) :: BX(n), BY(n), BZ(n)        
      real :: xgsw, ygsw, zgsw
      COMMON /GEOPACK1/ AA(10),SPS,CPS,BB(22)
      COMMON /GEOPACK2/ G(105),H(105),REC(105)

      SPS = 0.0
      CPS = 1.0
      
      do i = 1, n
         xgsw = X(i)
         ygsw = Y(i)
         zgsw = Z(i)
         DIPMOM=SQRT(G(2)**2+G(3)**2+H(3)**2)
         P=XGSW**2
         U=ZGSW**2
         V=3.*ZGSW*XGSW
         T=YGSW**2
         Q=DIPMOM/SQRT(P+T+U)**5
         BX(i)=Q*((T+U-2.*P)*SPS-V*CPS)
         BY(i)=-3.*YGSW*Q*(XGSW*SPS+ZGSW*CPS)
         BZ(i)=Q*((P+T-2.*U)*CPS-V*SPS)
c     all DIP_08(X(i),Y(i),Z(i),BX(i),BY(i),BZ(i))
      end do   
      end subroutine
