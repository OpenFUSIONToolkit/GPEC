c-----------------------------------------------------------------------
c     file fft.f.
c     fast fourier transform routines.
c     Numerical Recipes in Fortran, Second Edition, p. 501, four1.
c     Fortran 90 complex arithmetic version by Alan H. Glasser.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. fft_mod.
c     1. fft_run.
c     2. fft_write.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     subprogram 0. fft_mod.
c     module declarations.
c-----------------------------------------------------------------------
      module fft_mod
      use defs_mod
      implicit none

      contains
c-----------------------------------------------------------------------
c     subprogram 1. fft_run.
c     performs fast fourier transform.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      FUNCTION fft_run(f,sign) RESULT(g)

      complex(r8), dimension(:,:), intent(in) :: f
      complex(r8), dimension(SIZE(f,1),SIZE(f,2)) :: g
      integer, intent(in) :: sign

      integer :: i,istep,j,m,mmax,n
      complex(r8) :: w,wp
      complex(r8), dimension(SIZE(f,2)) :: temp
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      n=SIZE(f,1)
      select case(sign)
      case(-1)
         g=f/n
      case(1)
         g=f
      case default
         write(*,'(a,i2,a)')"sign = ",sign," is illegal, must be +/- 1"
         stop
      end select
c-----------------------------------------------------------------------
c     bit reversal.
c-----------------------------------------------------------------------
      j=1
      do i=1,n
         if(j > i)then
            temp(:)=g(j,:)
            g(j,:)=g(i,:)
            g(i,:)=temp
         endif
         m=n/2
         do
            if(m < 1 .OR. j <= m)EXIT
            j=j-m
            m=m/2
         enddo
         j=j+m
      enddo
c-----------------------------------------------------------------------
c     Danielson-Lanczos loop.
c-----------------------------------------------------------------------
      mmax=1
      do
         if(n <= mmax)EXIT
         istep=2*mmax
         wp=EXP(pi*ifac/(sign*mmax))
         w=1
         do m=1,mmax
            do i=m,n,istep
               j=i+mmax
               temp=w*g(j,:)
               g(j,:)=g(i,:)-temp
               g(i,:)=g(i,:)+temp
            enddo
            w=w*wp
         enddo
         mmax=istep
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end FUNCTION fft_run
c-----------------------------------------------------------------------
c     subprogram 2. fft_write.
c     ascii and binary output of fast fourier transform.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fft_write(t,f,g,h,out,bin,out_unit,bin_unit)

      real(r8), dimension(:), intent(in) :: t
      complex(r8), dimension(:,:), intent(in) :: f,g,h
      logical :: out,bin
      integer :: out_unit,bin_unit

      integer :: i,k,n
      real(r8) :: dt,tperiod,dfreq,freqmax

      integer, dimension(SIZE(t)) :: m
      real(r8), dimension(SIZE(t)) :: freq
      complex(r8), dimension(SIZE(g,1),SIZE(g,2)) :: gg
c-----------------------------------------------------------------------
c     format statements.
c-----------------------------------------------------------------------
 10   format(/6x,"n",7x,"dt",6x,"tperiod",5x,"dfreq",5x,"freqmax"
     $     //i8,1p,4e11.3/)
 20   format(/7x,"i",6x,"m",7x,"t",9x,"freq",7x,"re f",7x,"im f",7x,
     $     "re g",7x,"im g",7x,"re h",7x,"im h"/)
 30   format(2i8,1p,8e11.3)
c-----------------------------------------------------------------------
c     compute scalars.
c-----------------------------------------------------------------------
      n=SIZE(t)
      dt=t(2)-t(1)
      tperiod=n*dt
      dfreq=1/(n*dt)
      freqmax=dfreq*n/2
c-----------------------------------------------------------------------
c     compute auxiliary arrays.
c-----------------------------------------------------------------------
      m(1:n/2-1)=(/(i,i=1-n/2,-1)/)
      m(n/2:n)=(/(i,i=0,n/2)/)
      freq=m*dfreq
      gg(1:n/2-1,:)=g(n/2+2:n,:)
      gg(n/2:n,:)=g(1:n/2+1,:)
c-----------------------------------------------------------------------
c     diagnose fft.
c-----------------------------------------------------------------------
      write(out_unit,10)n,dt,tperiod,dfreq,freqmax
      do k=1,SIZE(f,2)
         if(out)then
            write(out_unit,'(/a,i3)')"k = ",k
            write(out_unit,20)
         endif
         do i=1,n
            if(out)write(out_unit,30)i-1,m(i),t(i),freq(i),
     $           f(i,k),gg(i,k),h(i,k)
            if(bin)write(bin_unit)
     $           real(i-1,4),real(m(i),4),real(t(i),4),real(freq(i),4),
     $           real(real(f(i,k)),4),real(AIMAG(f(i,k)),4),
     $           real(real(gg(i,k)),4),real(AIMAG(gg(i,k)),4),
     $           real(real(h(i,k)),4),real(AIMAG(h(i,k)),4)
         enddo
         if(out)write(out_unit,20)
         if(bin)write(bin_unit)
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fft_write
      end module fft_mod
