! First export:
!     export FFLAGS='-O0 -fbacktrace -g -fcheck=all -fPIC'
!     export FC=gfortran
! run using: make && gfortran TESTVAC.f *.o -std=legacy -framework Accelerate -o testvac && ./testvac


      MODULE TESTVAC_MOD
      IMPLICIT NONE
      CONTAINS

c-----------------------------------------------------------------------
c     Test 1: Test the green function from vacuum_ma.f
c-----------------------------------------------------------------------
      SUBROUTINE test_green

      USE vacuum_mod
      USE vglobal_mod
      IMPLICIT NONE

      !INTEGER, PARAMETER :: r8 = SELECTED_REAL_KIND(15, 307)
      INTEGER, PARAMETER :: mthvac = 900
      INTEGER :: mtheta = 200
      LOGICAL :: complex_flag = .TRUE.
      REAL(r8) :: kernelsignin = 1.0_r8
      LOGICAL :: walflag = .FALSE.
      LOGICAL :: farwalflag = .TRUE.
      INTEGER, PARAMETER :: mpert = 20
      COMPLEX(r8), DIMENSION(mpert, mpert) :: wv
      REAL(r8), DIMENSION(2*(mthvac+5),mpert*2) :: grriio
      REAL(r8), DIMENSION(mthvac+5,4) :: xzptso

      ! REAL(r8) :: xs, zs, xt, zt, xtp, ztp
      ! INTEGER :: n
      CALL defglo(mthvac)

      xs = 2.1629189218489446_r8
      zs = 0.49099087247571593_r8
      xt = 1.4309251319336729_r8
      zt = -1.0909471147721690_r8
      xtp = -0.31581804571944494_r8
      ztp = -0.13298631469337879_r8
      n = 1

      CALL green
!       CALL mscvac(wv,mpert,mtheta,mthvac,complex_flag,kernelsignin,
!      $                   walflag, farwalflag, grriio, xzptso)

c-----------------------------------------------------------------------
c     Write a bunch of outputs:
c     bval
c     aval
c     aval0
c-----------------------------------------------------------------------

      WRITE(*,*) "bval = ", bval
      WRITE(*,*) "aval = ", aval
      WRITE(*,*) "aval0 = ", aval0

      END SUBROUTINE test_green
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     Test 2: test fourier and fourier inverse from vacuum_vac.f
c-----------------------------------------------------------------------
      SUBROUTINE test_fourier
      USE vglobal_mod
      USE vacuum_mod
      REAL(r8), DIMENSION(:,:), ALLOCATABLE :: gij, gil, cs, gll
      REAL(r8), DIMENSION(:), ALLOCATABLE :: x
      INTEGER :: mtheta=200, mthvac=900, mpert=34
      INTEGER :: i, j
      kernelsign=1
      ntsin0=mtheta+1
      nths0=mthvac
      nfm=mpert
      mtot=mpert
      call global_alloc(nths0,nfm,mtot,ntsin0)
      CALL defglo(900)

      lmin(1) = 1
      lmax(1) = nfm

      WRITE(*,*) nths, nths2, nfm, nfm2

      nths = 905
      nths2 = 1810
      nfm = 34
      nfm2 = 68

      ALLOCATE(gij(nths,nths), gil(nths2,nfm2), cs(nths,nfm))
      ALLOCATE(gll(nfm,nfm))
      ALLOCATE(x(nths))

      gij = 0.0_r8
      gil = 0.0_r8
      cs = 0.0_r8
      gll = 0.0_r8

      DO i=1, nths
         x(i) = 6.28/REAL(i,r8)
      ENDDO


      DO i=1, nths
         DO j=1, nths
            gij(i,j) = SIN(x(i) + x(j))
         ENDDO

         DO j=1, nfm
            cs(i,j) = SIN(j*x(i))**2
         ENDDO
      ENDDO

      CALL fouran(gij,gil,cs,0,34)

      WRITE(*,*) 'gil(1:5,1:5) matrix:'
      DO i = 1, 5
         WRITE(*,'(5(F12.6,2X))') (gil(109+i,33+j), j=1,5)
      END DO

      mth = nths
      dth = twopi / mth

      CALL foranv(gil,gll,cs,0,34)

      WRITE(*,*) 'gll(1:5,1:5) matrix:'
      DO i = 1, 5
            WRITE(*,'(5(F12.6,2X))') (gll(i,j), j=1,5)
      END DO

      END SUBROUTINE test_fourier
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     Test 3: test mscvac from vacuum_vac.f
c-----------------------------------------------------------------------
      SUBROUTINE test_mscvac
      USE vacuum_mod
      USE vglobal_mod
      IMPLICIT NONE

      !INTEGER, PARAMETER :: r8 = SELECTED_REAL_KIND(15, 307)
      INTEGER, PARAMETER :: mthvac = 512
      INTEGER :: mtheta = 200
      INTEGER :: i, j
      INTEGER :: complex_flag = 1
      REAL(r8) :: kernelsignin = 1.0_r8
      INTEGER :: walflag = 0
      INTEGER :: farwalflag = 1
      INTEGER, PARAMETER :: mpert = 34
      COMPLEX(r8), DIMENSION(mpert, mpert) :: wv
      REAL(r8), DIMENSION(2*(mthvac+5),mpert*2) :: grriio
      REAL(r8), DIMENSION(mthvac+5,4) :: xzptso

      CALL mscvac(wv,mpert,mtheta,mthvac,complex_flag,kernelsignin,
     $                   walflag, farwalflag, grriio, xzptso)

      WRITE(*,*) 'wv(1:5,1:5) matrix:'
      DO i = 1, 5
         WRITE(*,'(5("(",F10.6,",",F10.6,")",2X))') (wv(i,j), j=1,5)
      END DO

      END SUBROUTINE test_mscvac
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     Test 4: test vaccal from vacuum_vac.f
c-----------------------------------------------------------------------
      SUBROUTINE test_vaccal
      USE vglobal_mod
      USE vacuum_mod
      IMPLICIT NONE

      INTEGER :: mtheta, mthvac, mpert
      REAL(r8), DIMENSION(:,:), ALLOCATABLE :: gil, gll, cs
      REAL(r8), DIMENSION(:,:), ALLOCATABLE :: gij
      INTEGER :: i, j

      mtheta = 200
      mthvac = 900
      mpert  = 20

      ! Initialize global sizes
      kernelsign = 1
      ntsin0 = mtheta + 1
      nths0  = mthvac
      nfm    = mpert
      mtot   = mpert
      CALL global_alloc(nths0, nfm, mtot, ntsin0)
      CALL defglo(mthvac)

      ! Example lmin/lmax for Fourier indices
      lmin(1) = 1
      lmax(1) = nfm

      ! Update nths/nfm (consistent with above)
      nths  = mthvac + 5
      nths2 = 2 * nths
      nfm2  = 2 * nfm

      ! Allocate working arrays
      ALLOCATE(gij(nths, nths))
      ALLOCATE(gil(nths2, nfm2))
      ALLOCATE(cs(nths, nfm))
      ALLOCATE(gll(nfm, nfm))

      gij = 0.0_r8
      gil = 0.0_r8
      cs  = 0.0_r8
      gll = 0.0_r8

      ! Example test input data for gij/cs
      DO i = 1, nths
         DO j = 1, nths
            gij(i,j) = SIN(REAL(i+j, r8) * 0.01_r8)
         END DO
         DO j = 1, nfm
            cs(i,j) = COS(REAL(i*j, r8) * 0.01_r8)
         END DO
      END DO

      check1 = .TRUE.
      farwal = .FALSE.

      PRINT *, 'Check gij(1:3,1:3):'
      DO i = 1,3
         WRITE(*,'(3(F12.6,2X))') (gij(i,j), j=1,3)
      END DO

      PRINT *, 'Check cs(1:3,1:3):'
      DO i = 1,3
         WRITE(*,'(3(F12.6,2X))') (cs(i,j), j=1,3)
      END DO

      PRINT *, 'Calling vaccal...'
      CALL vaccal()
      PRINT *, 'vaccal completed successfully'

      ! Dump a small part of arrays to check Julia port
      WRITE(*,*) 'gil(1:3,1:3):'
      DO i=1,3
         WRITE(*,'(3(F12.6,2X))') (gil(i,j), j=1,3)
      END DO

      END SUBROUTINE test_vaccal

c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     Test 5: test kernel from vacuum_vac.f
c-----------------------------------------------------------------------
      SUBROUTINE test_kernel
      USE vglobal_mod
      USE vacuum_mod
      USE test_kernel_data
      IMPLICIT NONE

      INTERFACE
         SUBROUTINE kernel(xobs,zobs,xsce,zsce,grdgre,gren,
     $        j1,j2,isgn,iopw,iops,ischk)
            USE vglobal_mod, ONLY: r8
            REAL(r8), DIMENSION(:), INTENT(IN) :: xobs,zobs,xsce,zsce
            REAL(r8), DIMENSION(:,:), INTENT(OUT) :: grdgre
            REAL(r8), DIMENSION(:,:), INTENT(OUT), TARGET :: gren
            INTEGER, INTENT(IN) :: j1,j2,isgn,iopw,iops,ischk
         END SUBROUTINE kernel
      END INTERFACE

      ! REAL(r8), DIMENSION(517) :: xobs, zobs, xsce, zsce
      REAL(r8), DIMENSION(:, :), ALLOCATABLE :: grdgre
      REAL(r8), DIMENSION(:, :), TARGET, ALLOCATABLE :: gren
      ! INTEGER :: j1, j2, isgn, iopw, iops, ischk
      INTEGER :: mthvac = 512
      ! INTEGER :: nths
      ! INTEGER :: nths2
      INTEGER :: j, i
      INTEGER :: mtheta = 256
      INTEGER :: debug_out_unit = 7
      INTEGER :: debug_out_unit_2 = 8

      ntsin0 = mtheta + 1
      nths0  = mthvac
      nfm    = 34
      mtot   = nfm

      CALL global_alloc(nths0,nfm,mtot,ntsin0)
      farwal = .true.
      CALL defglo(mthvac)

      ! WRITE(*,*) "mth1 = ", mth1
      ! WRITE(*,*) "nths2 = ", nths2


      ALLOCATE(gren(1:mth1,1:mth1))
      ALLOCATE(grdgre(nths2,nths2))
      ! Write out the size of the allocated arrays
      WRITE(*,*) "gren size: ", SIZE(gren,1), SIZE(gren,2)
      WRITE(*,*) "grdgre size: ", SIZE(grdgre,1), SIZE(grdgre,2)

      CALL kernel(xobs, zobs, xsce, zsce, grdgre, gren,
     $            j1, j2, isgn, iopw, iops, ischk)

      WRITE(*,*) 'gren(1:5,1:5) matrix:'
      DO i = 1, 5
         WRITE(*,'(5(F12.9,2X))') (gren(i,j), j=1,5)
      END DO
      WRITE(*,*) 'grdgre(1:5,1:5) matrix:'
      DO i = 1, 5
         WRITE(*,'(5(F12.9,2X))') (grdgre(i,j), j=1,5)
      END DO
      ! Do the same thing but for the last 5x5 block
      WRITE(*,*) 'gren(last 5x5) matrix:'
      DO i = mth1-4, mth1
         WRITE(*,'(5(F12.9,2X))') (gren(i,j), j=mth1-4,mth1)
      END DO
      WRITE(*,*) 'grdgre(last 5x5) matrix:'
      DO i = nths2-4, nths2
         WRITE(*,'(5(F12.9,2X))') (grdgre(i,j), j=nths2-4,nths2)
      END DO

      ! Write grdgre to a file
      OPEN(unit=debug_out_unit, file='test_kernel_grdgre_output.txt',
     $     status='replace', action='write', form='formatted')
      ! WRITE(debug_out_unit,*) 'grdgre matrix:'
      DO i = 1, SIZE(grdgre,1)
         WRITE(debug_out_unit,'(1034(F12.9,2X))') (grdgre(i,j),
     $                                            j=1,SIZE(grdgre,2))
      END DO
      CLOSE(debug_out_unit)

      ! Write gren to a file
      OPEN(unit=debug_out_unit_2, file='test_kernel_gren_output.txt',
     $     status='replace', action='write', form='formatted')
      ! WRITE(debug_out_unit_2,*) 'gren matrix:'
      DO i = 1, SIZE(gren,1)
         WRITE(debug_out_unit_2,'(514(F12.9,2X))') (gren(i,j),
     $                                            j=1,SIZE(gren,2))
      END DO
      CLOSE(debug_out_unit_2)

      END SUBROUTINE test_kernel
c-----------------------------------------------------------------------
      END MODULE TESTVAC_MOD

c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     Main program to run the tests
c-----------------------------------------------------------------------

      PROGRAM TESTVAC

      USE TESTVAC_MOD

      WRITE(*,*) "Running tests for vacuum module."
      ! WRITE(*,*) "-----------------------------------"
      ! WRITE(*,*) "Test 1: Green function"
      ! CALL test_green
      ! WRITE(*,*) "-----------------------------------"
      ! WRITE(*,*) "Test 2: Fourier and Inverse Fourier"
      ! CALL test_fourier
      ! WRITE(*,*) "-----------------------------------"
      ! WRITE(*,*) "Test 4: Vaccal"
      ! CALL test_vaccal
      WRITE(*,*) "-----------------------------------"
      WRITE(*,*) "Test 5: Kernel"
      CALL test_kernel
      ! WRITE(*,*) "-----------------------------------"
      ! WRITE(*,*) "Last Test: MSCVAC"
      ! CALL test_mscvac

      WRITE(*,*) "-----------------------------------"
      WRITE(*,*) "All tests completed successfully."

      END PROGRAM TESTVAC
