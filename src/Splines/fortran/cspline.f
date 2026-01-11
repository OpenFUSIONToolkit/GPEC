c-----------------------------------------------------------------------
c     file cspline.f
c     fits complex functions to cubic splines.
c     Reference: H. Spaeth, "Spline Algorithms for Curves and Surfaces,"
c     Translated from the German by W. D. Hoskins and H. W. Sager.
c     Utilitas Mathematica Publishing Inc., Winnepeg, 1974.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. cspline_mod.
c     1. cspline_alloc.
c     2. cspline_dealloc.
c     3. cspline_fit.
c     4. cspline_fit_classic.
c     5. cspline_fit_ahg.
c     6. cspline_fac.
c     7. cspline_eval.
c     8. cspline_eval_external.
c     9. cspline_all_eval.
c     10. cspline_write.
c     11. cspline_write_log.
c     12. cspline_int.
c     13. cspline_triluf.
c     14. cspline_trilus.
c     15. cspline_sherman.
c     16. cspline_morrison.
c     17. cspline_copy.
c     18. cspline_thomas.
c     19. cspline_get_yp.
c-----------------------------------------------------------------------
c     subprogram 0. cspline_type definition.
c     defines cspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      module cspline_mod
      use spline_mod
      implicit none

      type :: cspline_type
      integer :: mx,nqty,ix
      real(r8), dimension(:), POinTER :: xs
      real(r8), dimension(2) :: x0
      real(r8), dimension(:,:), POinTER :: xpower
      complex(r8), dimension(:), POinTER :: f,f1,f2,f3
      complex(r8), dimension(:,:), POinTER :: fs,fs1,fsi
      character(6), dimension(:), POinTER :: title
      character(6) :: name
      logical :: periodic
      logical :: allocated=.false.
      end type cspline_type

      contains
c-----------------------------------------------------------------------
c     subprogram 1. cspline_alloc.
c     allocates space for cspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_alloc(spl,mx,nqty)

      integer, intent(in) :: mx,nqty
      type(cspline_type), intent(inout) :: spl

c-----------------------------------------------------------------------
c     safety check.
c-----------------------------------------------------------------------
      if(spl%allocated)then
         call program_stop("cspline_alloc: already allocated")
      endif

c-----------------------------------------------------------------------
c     set scalars.
c-----------------------------------------------------------------------
      spl%mx=mx
      spl%nqty=nqty
      spl%ix=0
      spl%periodic=.false.
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      allocate(spl%xs(0:mx))
      allocate(spl%f(nqty))
      allocate(spl%f1(nqty))
      allocate(spl%f2(nqty))
      allocate(spl%f3(nqty))
      allocate(spl%title(0:nqty))
      allocate(spl%fs(0:mx,nqty))
      allocate(spl%fs1(0:mx,nqty))
      allocate(spl%xpower(2,nqty))
      spl%xpower=0
      spl%x0=0
      NULLifY(spl%fsi)
      spl%allocated=.true.
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_alloc
c-----------------------------------------------------------------------
c     subprogram 2. cspline_dealloc.
c     deallocates space for cspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_dealloc(spl)

      type(cspline_type), intent(inout) :: spl
c-----------------------------------------------------------------------
c     safety check.
c-----------------------------------------------------------------------
      if(.not.spl%allocated)then
         call program_stop("cspline_dealloc: not allocated")
      endif


c-----------------------------------------------------------------------
c     deallocate space.
c-----------------------------------------------------------------------

      deallocate(spl%xs)
      deallocate(spl%f)
      deallocate(spl%f1)
      deallocate(spl%f2)
      deallocate(spl%f3)
      deallocate(spl%title)
      deallocate(spl%fs)
      deallocate(spl%fs1)
      deallocate(spl%xpower)
      if(ASSOCIATED(spl%fsi))deallocate(spl%fsi)
      spl%allocated=.false.
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_dealloc
c-----------------------------------------------------------------------
c     subprogram 3. cspline_fit.
c     router between Glasser and classic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_fit(spl,endmode)

      type(cspline_type), intent(inout) :: spl
      integer, intent(in) :: endmode
c-----------------------------------------------------------------------
c     switch between csplines.
c-----------------------------------------------------------------------
c      - use_classic_splines is always False
c      if (use_classic_splines .and.
c     $    (endmode == 3 .OR. endmode == 1))then ! 3 = Extrapolate, 1= natural
c         call cspline_fit_classic(spl,endmode)
c      else
         call cspline_fit_ahg(spl,endmode)
c      endif

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_fit
c-----------------------------------------------------------------------
c     subprogram 4. cspline_fit_classic.
c     classical spline solution as in spline.f, but now with complex r
c     does not support periodic and not-a-knot boundary conditions
c     g=a+b(x-xi)+c(x-xi)^2+d(x-xi)^3
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_fit_classic(spl,endmode)
      type(cspline_type), intent(inout) :: spl
      integer, intent(in) :: endmode
      real(r8), dimension(:),allocatable :: d,l,u,h
      complex(r8), dimension(:,:),allocatable :: r
      real(r8), dimension(0:spl%mx) :: xfac

      integer :: iside,iqty,i
      complex(r8),dimension(spl%nqty) :: bs,cs,ds
c-----------------------------------------------------------------------
c     extract powers.
c-----------------------------------------------------------------------
      do iside=1,2
         do iqty=1,spl%nqty
            if(spl%xpower(iside,iqty) /= 0)then
               xfac=1/ABS(spl%xs-spl%x0(iside))**spl%xpower(iside,iqty)
               spl%fs(:,iqty)=spl%fs(:,iqty)*xfac
            endif
         enddo
      enddo
      allocate (d(0:spl%mx),l(spl%mx),u(spl%mx),r(0:spl%mx,spl%nqty))
      allocate (h(0:spl%mx-1))
c-----------------------------------------------------------------------
c     compute tridiagnol matrix for natural B.C.
c-----------------------------------------------------------------------
      do i=0,spl%mx-1
         h(i)=spl%xs(i+1)-spl%xs(i)
      enddo

      d(0)=1
      do i=1,spl%mx-1
         d(i)=2*(h(i-1)+h(i))
      enddo
      d(spl%mx)=1

      do i=1,spl%mx-1
         l(i)=h(i-1)
      enddo
      l(spl%mx)=0

      u(1)=0
      do i=2,spl%mx
         u(i)=h(i-1)
      enddo

      r(0,:)=0
      do i=1,spl%mx-1
         r(i,:)=( (spl%fs(i+1,:)-spl%fs(i,:))/h(i)
     $           -(spl%fs(i,:)-spl%fs(i-1,:))/h(i-1) )*six
      enddo
      r(spl%mx,:)=0

      if (endmode==3) then ! 3 = Extrapolated
         call cspline_get_yp(spl%xs(0:3),spl%fs(0:3,:),
     $                      spl%xs(0),r(0,:),spl%nqty)
         call cspline_get_yp(spl%xs(spl%mx-3:spl%mx),
     $        spl%fs(spl%mx-3:spl%mx,:),spl%xs(spl%mx),
     $        r(spl%mx,:),spl%nqty)
         d(0)=2*h(0)
         d(spl%mx)=2*h(spl%mx-1)
         u(1)=h(0)
         l(spl%mx)=h(spl%mx-1)
         r(0,:)=( (spl%fs(1,:)-spl%fs(0,:))/h(0) - r(0,:) )*six
         r(spl%mx,:)=( r(spl%mx,:)
     $  -(spl%fs(spl%mx,:)-spl%fs(spl%mx-1,:))/h(spl%mx-1) )*six

      endif
c-----------------------------------------------------------------------
c     solve and contrruct spline.
c-----------------------------------------------------------------------

      call cspline_thomas(l,d,u,r,spl%mx+1,spl%nqty)

      do i=0, spl%mx-1
         bs=(spl%fs(i+1,:)-spl%fs(i,:))/h(i)
     $    - half*h(i)*r(i,:)
     $    - h(i)*(r(i+1,:)-r(i,:))/six
         spl%fs1(i,:)=bs
      enddo
      ds=(r(spl%mx,:)-r(spl%mx-1,:))/(h(spl%mx-1)*six)
      cs=r(spl%mx-1,:)*half
      i=spl%mx-1
      bs=(spl%fs(i+1,:)-spl%fs(i,:))/h(i)
     $    - half*h(i)*r(i,:)
     $    - h(i)*(r(i+1,:)-r(i,:))/six
      i=spl%mx
      spl%fs1(i,:)=bs+h(i-1)*(cs*2+h(i-1)*ds*3)
      deallocate (d,l,u,r,h)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_fit_classic
c-----------------------------------------------------------------------
c     subprogram 5. cspline_fit_ahg.
c     fits complex functions to cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_fit_ahg(spl,endmode)

      type(cspline_type), intent(inout) :: spl
      integer, intent(in) :: endmode

      integer :: iqty,iside
      real(r8), dimension(-1:1,0:spl%mx) :: a
      real(r8), dimension(spl%mx) :: b
      real(r8), dimension(4) :: cl,cr
      real(r8), dimension(0:spl%mx) :: xfac
c-----------------------------------------------------------------------
c     extract powers.
c-----------------------------------------------------------------------
      do iside=1,2
         do iqty=1,spl%nqty
            if(spl%xpower(iside,iqty) /= 0)then
               xfac=1/ABS(spl%xs-spl%x0(iside))**spl%xpower(iside,iqty)
               spl%fs(:,iqty)=spl%fs(:,iqty)*xfac
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     set up grid matrix.
c-----------------------------------------------------------------------
      call cspline_fac(spl,a,b,cl,cr,endmode)
c-----------------------------------------------------------------------
c     compute first derivatives, interior.
c-----------------------------------------------------------------------
      do iqty=1,spl%nqty
         spl%fs1(1:spl%mx-1,iqty)=
     $        3*((spl%fs(2:spl%mx,iqty)-spl%fs(1:spl%mx-1,iqty))
     $        *b(2:spl%mx)
     $        +(spl%fs(1:spl%mx-1,iqty)-spl%fs(0:spl%mx-2,iqty))
     $        *b(1:spl%mx-1))
      enddo
c-----------------------------------------------------------------------
c     extrapolation boundary conditions.
c-----------------------------------------------------------------------
      select case(endmode)
      case(3) ! 3 = Extrapolate
         do iqty=1,spl%nqty
            spl%fs1(0,iqty)=SUM(cl(1:4)*spl%fs(0:3,iqty))
            spl%fs1(spl%mx,iqty)=SUM(cr(1:4)
     $           *spl%fs(spl%mx:spl%mx-3:-1,iqty))
            spl%fs1(1,iqty)=spl%fs1(1,iqty)-spl%fs1(0,iqty)
     $           /(spl%xs(1)-spl%xs(0))
            spl%fs1(spl%mx-1,iqty)=
     $           spl%fs1(spl%mx-1,iqty)-spl%fs1(spl%mx,iqty)
     $           /(spl%xs(spl%mx)-spl%xs(spl%mx-1))
         enddo
         call cspline_trilus(a(:,1:spl%mx-1),spl%fs1(1:spl%mx-1,:))
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      case(4) ! 4 = not-a-knot
         spl%fs1(1,:)=spl%fs1(1,:)-(2*spl%fs(1,:)
     $        -spl%fs(0,:)-spl%fs(2,:))*2*b(1)
         spl%fs1(spl%mx-1,:)=spl%fs1(spl%mx-1,:)
     $        +(2*spl%fs(spl%mx-1,:)-spl%fs(spl%mx,:)
     $        -spl%fs(spl%mx-2,:))*2*b(spl%mx)
         call cspline_trilus(a(:,1:spl%mx-1),spl%fs1(1:spl%mx-1,:))
         spl%fs1(0,:)=(2*(2*spl%fs(1,:)-spl%fs(0,:)-spl%fs(2,:))
     $        +(spl%fs1(1,:)+spl%fs1(2,:))*(spl%xs(2)-spl%xs(1))
     $        -spl%fs1(1,:)*(spl%xs(1)-spl%xs(0)))/(spl%xs(1)-spl%xs(0))
         spl%fs1(spl%mx,:)=
     $        (2*(spl%fs(spl%mx-2,:)+spl%fs(spl%mx,:)
     $        -2*spl%fs(spl%mx-1,:))
     $        +(spl%fs1(spl%mx-1,:)+spl%fs1(spl%mx-2,:))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        -spl%fs1(spl%mx-1,:)
     $        *(spl%xs(spl%mx)-spl%xs(spl%mx-1)))
     $        /(spl%xs(spl%mx)-spl%xs(spl%mx-1))
c-----------------------------------------------------------------------
c     periodic boundary conditions.
c-----------------------------------------------------------------------
      case(2) ! 2 = Periodic
         spl%periodic=.true.
         spl%fs1(0,:)=3*((spl%fs(1,:)-spl%fs(0,:))*b(1)
     $        +(spl%fs(0,:)-spl%fs(spl%mx-1,:))*b(spl%mx))
         call cspline_morrison(a(:,0:spl%mx-1),spl%fs1(0:spl%mx-1,:))
         spl%fs1(spl%mx,:)=spl%fs1(0,:)
c-----------------------------------------------------------------------
c     unrecognized boundary condition.
c-----------------------------------------------------------------------
      case default
         call program_stop("Cannot recognize endmode")
      end select
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_fit_ahg
c-----------------------------------------------------------------------
c     subprogram 6. cspline_fac.
c     sets up matrix for cubic spline fitting.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_fac(spl,a,b,cl,cr,endmode)

      type(cspline_type), intent(in) :: spl
      real(r8), dimension(-1:1,0:spl%mx), intent(out) :: a
      real(r8), dimension(spl%mx), intent(out) :: b
      real(r8), dimension(4), intent(out) :: cl,cr
      integer, intent(in) :: endmode

      integer :: j
c-----------------------------------------------------------------------
c     compute interior matrix.
c-----------------------------------------------------------------------
      b=1/(spl%xs(1:spl%mx)-spl%xs(0:spl%mx-1))
      do j=1,spl%mx-1
         a(-1,j)=b(j)
         a(0,j)=2*(b(j)+b(j+1))
         a(1,j)=b(j+1)
      enddo
c-----------------------------------------------------------------------
c     extrapolation boundary conditions.
c-----------------------------------------------------------------------
      select case(endmode)
      case(3) ! 3 = Extrapolate
         b=b*b
         cl(1)=(spl%xs(0)*(3*spl%xs(0)
     $        -2*(spl%xs(1)+spl%xs(2)+spl%xs(3)))
     $        +spl%xs(1)*spl%xs(2)+spl%xs(1)*spl%xs(3)
     $        +spl%xs(2)*spl%xs(3))
     $        /((spl%xs(0)-spl%xs(1))*(spl%xs(0)-spl%xs(2))
     $        *(spl%xs(0)-spl%xs(3)))
         cl(2)=((spl%xs(2)-spl%xs(0))*(spl%xs(3)-spl%xs(0)))
     $        /((spl%xs(1)-spl%xs(0))*(spl%xs(1)-spl%xs(2))
     $        *(spl%xs(1)-spl%xs(3)))
         cl(3)=((spl%xs(0)-spl%xs(1))*(spl%xs(3)-spl%xs(0)))
     $        /((spl%xs(0)-spl%xs(2))*(spl%xs(1)-spl%xs(2))
     $        *(spl%xs(3)-spl%xs(2)))
         cl(4)=((spl%xs(1)-spl%xs(0))*(spl%xs(2)-spl%xs(0)))
     $        /((spl%xs(3)-spl%xs(0))*(spl%xs(3)-spl%xs(1))
     $        *(spl%xs(3)-spl%xs(2)))
         cr(1)=(spl%xs(spl%mx)*(3*spl%xs(spl%mx)
     $        -2*(spl%xs(spl%mx-1)+spl%xs(spl%mx-2)
     $        +spl%xs(spl%mx-3)))+spl%xs(spl%mx-1)
     $        *spl%xs(spl%mx-2)+spl%xs(spl%mx-1)*spl%xs(spl%mx
     $        -3)+spl%xs(spl%mx-2)*spl%xs(spl%mx-3))
     $        /((spl%xs(spl%mx)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx)-spl%xs(spl%mx-2))*(spl%xs(spl%mx
     $        )-spl%xs(spl%mx-3)))
         cr(2)=((spl%xs(spl%mx-2)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx-1)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-3)))
         cr(3)=((spl%xs(spl%mx)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-2)))
         cr(4)=((spl%xs(spl%mx-1)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-2)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx-3)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-2)))
         call cspline_triluf(a(:,1:spl%mx-1))
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      case(4) ! 4 = not-a-knot
         b=b*b
         a(0,1)=a(0,1)+(spl%xs(2)+spl%xs(0)-2*spl%xs(1))*b(1)
         a(1,1)=a(1,1)+(spl%xs(2)-spl%xs(1))*b(1)
         a(0,spl%mx-1)=a(0,spl%mx-1)
     $        +(2*spl%xs(spl%mx-1)-spl%xs(spl%mx-2)
     $        -spl%xs(spl%mx))*b(spl%mx)
         a(-1,spl%mx-1)=a(-1,spl%mx-1)
     $        +(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))*b(spl%mx)
         call cspline_triluf(a(:,1:spl%mx-1))
c-----------------------------------------------------------------------
c     periodic boundary conditions.
c-----------------------------------------------------------------------
      case(2) ! 2 = Periodic
         a(0,0:spl%mx:spl%mx)=2*(b(spl%mx)+b(1))
         a(1,0)=b(1)
         a(-1,0)=b(spl%mx)
         b=b*b
         call cspline_sherman(a(:,0:spl%mx-1))
c-----------------------------------------------------------------------
c     unrecognized boundary condition.
c-----------------------------------------------------------------------
      case default
         call program_stop("Cannot recognize endmode")
      end select
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_fac
c-----------------------------------------------------------------------
c     subprogram 7. cspline_eval.
c     evaluates complex cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_eval(spl,x,mode)

      type(cspline_type), intent(inout) :: spl
      real(r8), intent(in) :: x
      integer, intent(in) :: mode

      integer :: iqty,iside
      real(r8) :: xx,d,z,z1,xfac,dx
      complex(r8) :: g,g1,g2,g3
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      xx=x
      spl%ix=MAX(spl%ix,0)
      spl%ix=Min(spl%ix,spl%mx-1)
c-----------------------------------------------------------------------
c     normalize interval for periodic splines.
c-----------------------------------------------------------------------
      if(spl%periodic)then
         do
            if(xx < spl%xs(spl%mx))EXIT
            xx=xx-spl%xs(spl%mx)
         enddo
         do
            if(xx >= spl%xs(0))EXIT
            xx=xx+spl%xs(spl%mx)
         enddo
      endif
c-----------------------------------------------------------------------
c     find cubic spline interval.
c-----------------------------------------------------------------------
      do
         if(spl%ix <= 0)EXIT
         if(xx >= spl%xs(spl%ix))EXIT
         spl%ix=spl%ix-1
      enddo
      do
         if(spl%ix >= spl%mx-1)EXIT
         if(xx < spl%xs(spl%ix+1))EXIT
         spl%ix=spl%ix+1
      enddo
c-----------------------------------------------------------------------
c     evaluate offset and related quantities.
c-----------------------------------------------------------------------
      d=spl%xs(spl%ix+1)-spl%xs(spl%ix)
      z=(xx-spl%xs(spl%ix))/d
      z1=1-z
c-----------------------------------------------------------------------
c     evaluate functions.
c-----------------------------------------------------------------------
      spl%f=spl%fs(spl%ix,:)*z1*z1*(3-2*z1)
     $     +spl%fs(spl%ix+1,:)*z*z*(3-2*z)
     $     +d*z*z1*(spl%fs1(spl%ix,:)*z1
     $     -spl%fs1(spl%ix+1,:)*z)
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      if(mode > 0)then
         spl%f1=6*(spl%fs(spl%ix+1,:)
     $        -spl%fs(spl%ix,:))*z*z1/d
     $        +spl%fs1(spl%ix,:)*z1*(3*z1-2)
     $        +spl%fs1(spl%ix+1,:)*z*(3*z-2)
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      if(mode > 1)then
         spl%f2=(6*(spl%fs(spl%ix+1,:)
     $        -spl%fs(spl%ix,:))*(z1-z)/d
     $        -spl%fs1(spl%ix,:)*(6*z1-2)
     $        +spl%fs1(spl%ix+1,:)*(6*z-2))/d
      endif
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      if(mode > 2)then
         spl%f3=(12*(spl%fs(spl%ix,:)
     $        -spl%fs(spl%ix+1,:))/d
     $        +6*(spl%fs1(spl%ix,:)
     $        +spl%fs1(spl%ix+1,:)))/(d*d)
      endif
c-----------------------------------------------------------------------
c     restore powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=x-spl%x0(iside)
         do iqty=1,spl%nqty
            if(spl%xpower(iside,iqty) == 0)cycle
            xfac=ABS(dx)**spl%xpower(iside,iqty)
            g=spl%f(iqty)*xfac
            if(mode > 0)g1=(spl%f1(iqty)+spl%f(iqty)
     $           *spl%xpower(iside,iqty)/dx)*xfac
            if(mode > 1)g2=(spl%f2(iqty)+spl%xpower(iside,iqty)/dx
     $           *(2*spl%f1(iqty)+(spl%xpower(iside,iqty)-1)
     $           *spl%f(iqty)/dx))*xfac
            if(mode > 2)g3=(spl%f3(iqty)+spl%xpower(iside,iqty)/dx
     $           *(3*spl%f2(iqty)+(spl%xpower(iside,iqty)-1)/dx
     $           *(3*spl%f1(iqty)+(spl%xpower(iside,iqty)-2)/dx
     $           *spl%f(iqty))))*xfac
            spl%f(iqty)=g
            if(mode > 0)spl%f1(iqty)=g1
            if(mode > 1)spl%f2(iqty)=g2
            if(mode > 2)spl%f3(iqty)=g3
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_eval
c-----------------------------------------------------------------------
c     subprogram 8. cspline_eval_external.
c     evaluates complex cubic splines with external arrays (parallel).
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_eval_external(spl,x,s_f,s_f1,s_f2,s_f3)

      type(cspline_type), intent(in) :: spl
      real(r8), intent(in) :: x

      integer :: iqty,iside
      integer :: ix, i_low, i_high, i_mid
      real(r8) :: xx,d,z,z1,dx
      complex(r8) :: g,g1,g2,g3

      complex(r8), dimension(:), intent(inout) :: s_f
      complex(r8), dimension(:),optional,intent(out) :: s_f1,s_f2,s_f3

      real(r8) :: xpow,xfac
c-----------------------------------------------------------------------
c     zero out external array.
c-----------------------------------------------------------------------
      s_f = 0
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      xx=x
c-----------------------------------------------------------------------
c     normalize interval for periodic splines.
c-----------------------------------------------------------------------
      if(spl%periodic)then
         do
            if(xx < spl%xs(spl%mx))EXIT
            xx=xx-spl%xs(spl%mx)
         enddo
         do
            if(xx >= spl%xs(0))EXIT
            xx=xx+spl%xs(spl%mx)
         enddo
      endif
c-----------------------------------------------------------------------
c     find cubic spline interval using Binary Search
c-----------------------------------------------------------------------
      i_low = 0
      i_high = spl%mx - 1

      do while (i_low <= i_high)
            i_mid = i_low + (i_high - i_low) / 2
            if (xx < spl%xs(i_mid)) then
                  i_high = i_mid - 1
            else if (xx >= spl%xs(i_mid + 1)) then
                  i_low = i_mid + 1
            else
                  ! We found the interval: xs(i_mid) <= xx < xs(i_mid+1)
                  ix = i_mid
                  exit
            endif
      enddo

      ! If the search fails, treat it as a boundary value.
      if (i_low > i_high) then
        ix = min(max(i_low - 1, 0), spl%mx - 1)
      endif
c-----------------------------------------------------------------------
c     evaluate offset and related quantities.
c-----------------------------------------------------------------------
      d=spl%xs(ix+1)-spl%xs(ix)
      z=(xx-spl%xs(ix))/d
      z1=1-z
c-----------------------------------------------------------------------
c     evaluate functions.
c-----------------------------------------------------------------------
      s_f=spl%fs(ix,:)*z1*z1*(3-2*z1)
     $     +spl%fs(ix+1,:)*z*z*(3-2*z)
     $     +d*z*z1*(spl%fs1(ix,:)*z1
     $     -spl%fs1(ix+1,:)*z)
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      if(present(s_f1))then
         s_f1=6*(spl%fs(ix+1,:)
     $        -spl%fs(ix,:))*z*z1/d
     $        +spl%fs1(ix,:)*z1*(3*z1-2)
     $        +spl%fs1(ix+1,:)*z*(3*z-2)
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      if(present(s_f2))then
         s_f2=(6*(spl%fs(ix+1,:)
     $        -spl%fs(ix,:))*(z1-z)/d
     $        -spl%fs1(ix,:)*(6*z1-2)
     $        +spl%fs1(ix+1,:)*(6*z-2))/d
      endif
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      if(present(s_f3))then
         s_f3=(12*(spl%fs(ix,:)
     $        -spl%fs(ix+1,:))/d
     $        +6*(spl%fs1(ix,:)
     $        +spl%fs1(ix+1,:)))/(d*d)
      endif
c-----------------------------------------------------------------------
c     restore powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=ABS(x-spl%x0(iside))
         do iqty=1,spl%nqty
            xpow = spl%xpower(iside,iqty)
            if(xpow == 0)cycle
            xfac=dx**xpow
            g=s_f(iqty)*xfac
            if(present(s_f1))g1=(s_f1(iqty)+s_f(iqty)
     $           *spl%xpower(iside,iqty)/dx)*xfac
            if(present(s_f2))g2=(s_f2(iqty)+spl%xpower(iside,iqty)/dx
     $           *(2*s_f1(iqty)+(spl%xpower(iside,iqty)-1)
     $           *s_f(iqty)/dx))*xfac
            if(present(s_f3))g3=(s_f3(iqty)+spl%xpower(iside,iqty)/dx
     $           *(3*s_f2(iqty)+(spl%xpower(iside,iqty)-1)/dx
     $           *(3*s_f1(iqty)+(spl%xpower(iside,iqty)-2)/dx
     $           *s_f(iqty))))*xfac
            s_f(iqty)=g
            if(present(s_f1))s_f1(iqty)=g1
            if(present(s_f2))s_f2(iqty)=g2
            if(present(s_f3))s_f3(iqty)=g3
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_eval_external
c-----------------------------------------------------------------------
c     subprogram 9. cspline_all_eval.
c     evaluates cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_all_eval(spl,z,f,f1,f2,f3,mode)

      type(cspline_type), intent(inout) :: spl
      real(r8), intent(in) :: z
      complex(r8), dimension(spl%mx,spl%nqty), intent(out) ::
     $     f,f1,f2,f3
      integer, intent(in) :: mode

      integer :: iqty,nqty,n,iside
      real(r8) :: z1
      real(r8), dimension(spl%mx) :: d,xfac,dx
      complex(r8), dimension(spl%mx) :: g,g1,g2,g3
c-----------------------------------------------------------------------
c     evaluate offset and related quantities.
c-----------------------------------------------------------------------
      n=spl%mx
      nqty=spl%nqty
      z1=1-z
      d=spl%xs(1:n)-spl%xs(0:n-1)
c-----------------------------------------------------------------------
c     evaluate functions.
c-----------------------------------------------------------------------
      do iqty=1,nqty
         f(:,iqty)=spl%fs(0:n-1,iqty)*z1*z1*(3-2*z1)
     $        +spl%fs(1:n,iqty)*z*z*(3-2*z)
     $        +d*z*z1*(spl%fs1(0:n-1,iqty)*z1-spl%fs1(1:n,iqty)*z)
      enddo
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      if(mode > 0)then
         do iqty=1,nqty
            f1(:,iqty)=6*(spl%fs(1:n,iqty)-spl%fs(0:n-1,iqty))*z*z1/d
     $           +spl%fs1(0:n-1,iqty)*z1*(3*z1-2)
     $           +spl%fs1(1:n,iqty)*z*(3*z-2)
         enddo
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      if(mode > 1)then
         do iqty=1,nqty
            f2(:,iqty)=(6*(spl%fs(1:n,iqty)-spl%fs(0:n-1,iqty))*(z1-z)/d
     $           -spl%fs1(0:n-1,iqty)*(6*z1-2)
     $           +spl%fs1(1:n,iqty)*(6*z-2))/d
         enddo
      endif
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      if(mode > 2)then
         do iqty=1,nqty
            f3(:,iqty)=(12*(spl%fs(0:n-1,iqty)-spl%fs(1:n,iqty))/d
     $           +6*(spl%fs1(0:n-1,iqty)+spl%fs1(1:n,iqty)))/(d*d)
         enddo
      endif
c-----------------------------------------------------------------------
c     restore powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=(spl%xs(0:spl%mx-1)+z*d(1:spl%mx))-spl%x0(iside)
         do iqty=1,spl%nqty
            if(spl%xpower(iside,iqty) == 0)cycle
            xfac=ABS(dx)**spl%xpower(iside,iqty)
            g=f(:,iqty)*xfac
            if(mode > 0)g1=(f1(:,iqty)
     $           +f(:,iqty)*spl%xpower(iside,iqty)/dx)*xfac
            if(mode > 1)g2=(f2(:,iqty)+spl%xpower(iside,iqty)/dx
     $           *(2*f1(:,iqty)+(spl%xpower(iside,iqty)-1)
     $           *f(:,iqty)/dx))*xfac
     $
            if(mode > 2)g3=(f3(:,iqty)+spl%xpower(iside,iqty)/dx
     $           *(3*f2(:,iqty)+(spl%xpower(iside,iqty)-1)/dx
     $           *(3*f1(:,iqty)+(spl%xpower(iside,iqty)-2)/dx
     $           *f(:,iqty))))*xfac
            f(:,iqty)=g
            if(mode > 0)f1(:,iqty)=g1
            if(mode > 1)f2(:,iqty)=g2
            if(mode > 2)f3(:,iqty)=g3
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_all_eval
c-----------------------------------------------------------------------
c     subprogram 10. cspline_write.
c     produces ascii and binary output.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_write(spl,out,bin,iua,iub,interp)

      type(cspline_type), intent(inout) :: spl
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      logical, intent(in) :: interp

      character(80) :: format1,format2
      integer :: i,j
      real(r8) :: x,dx
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   format('(/4x,"i",4x,a6,1x,',i2.2,'(2x,"re ",a6,2x,"im ",a6)/)')
 20   format('(i5,1p,',i2.2,'e11.3)')
!  30   format('(/4x,"i",2x,"j",',i2.2,'(4x,a6,1x)/)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not.out.and..not.bin)return
c-----------------------------------------------------------------------
c     print ascii tables of node values and derivatives.
c-----------------------------------------------------------------------
      if(out)then
         write(format1,10)spl%nqty
         write(format2,20)2*spl%nqty+1
         write(iua,'(/1x,a)')'node values:'
         write(iua,format1)spl%title(0),
     $        (spl%title(i),spl%title(i),i=1,spl%nqty)
      endif
      do i=0,spl%mx
         call cspline_eval(spl,spl%xs(i),0)
         if(out)write(iua,format2)spl%xs(i),spl%f
         if(bin)write(iub)real(spl%xs(i),4),
     $        real(spl%f,4),real(AIMAG(spl%f),4)
      enddo
      if(out)write(iua,format1)spl%title(0),
     $     (spl%title(i),spl%title(i),i=1,spl%nqty)
      if(bin)write(iub)
      if(.not. interp)return
c-----------------------------------------------------------------------
c     print header for interpolated values.
c-----------------------------------------------------------------------
      if(out)then
         write(iua,'(/1x,a)')'interpolated values:'
         write(iua,format1)spl%title(0),
     $        (spl%title(i),spl%title(i),i=1,spl%nqty)
      endif
c-----------------------------------------------------------------------
c     print interpolated values.
c-----------------------------------------------------------------------
      do i=0,spl%mx-1
         dx=(spl%xs(i+1)-spl%xs(i))/4
         do j=0,4
            x=spl%xs(i)+j*dx
            call cspline_eval(spl,x,0)
            if(out)write(iua,format2)i,x,spl%f
            if(bin)write(iub)real(x,4),
     $           real(spl%f,4),real(AIMAG(spl%f),4)
         enddo
      enddo
c-----------------------------------------------------------------------
c     print final interpolated values.
c-----------------------------------------------------------------------
      x=spl%xs(spl%mx)
      call cspline_eval(spl,x,0)
      if(out)then
         write(iua,format2)i,x,spl%f
         write(iua,format1)spl%title(0),
     $        (spl%title(i),spl%title(i),i=1,spl%nqty)
      endif
      if(bin)then
         write(iub)real(x,4),
     $        real(spl%f,4),real(AIMAG(spl%f),4)
         write(iub)
         call bin_close(bin_unit)
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_write
c-----------------------------------------------------------------------
c     subprogram 11. cspline_write_log
c     produces ascii and binary output of logs.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_write_log(spl,out,bin,iua,iub,interp,stride,
     $     xend)

      type(cspline_type), intent(inout) :: spl
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      integer, intent(in) :: stride
      logical, intent(in) :: interp
      real(r8), dimension(2) :: xend

      character(50) :: format1,format2
      integer :: iqty,ix,j,offset
      real(r8), PARAMETER :: epsilon=-20*alog10
      real(r8) :: x,dx
      real(r8), dimension(2) :: xlog
      complex(r8), dimension(spl%nqty/stride) :: flog
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   format('(/4x,"i",4x,a6,1x,',i2.2,'(2x,"re ",a6,2x,"im ",a6)/)')
 20   format('(i5,1p,',i2.2,'e11.3)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not. out .and. .not. bin)return
      if(MOD(spl%nqty,stride) /= 0)then
         write(*,'(2(a,i3))')"Cspline_write_log: nqty = ",spl%nqty,
     $        " is not an integral multiple of stride = ",stride
         stop
      endif
c-----------------------------------------------------------------------
c     write title and start loop over offsets.
c-----------------------------------------------------------------------
      if(out)write(iua,'(1x,a)')
     $     "Output from cspline_write_log for "//TRIM(spl%name)//":"
      do offset=1,stride
c-----------------------------------------------------------------------
c     print ascii table of node values.
c-----------------------------------------------------------------------
         if(out)then
            write(format1,10)spl%nqty/stride
            write(format2,20)2*spl%nqty/stride+1
            write(iua,'(/1x,a,i3,a)')
     $           "input values for offset = ",offset-1,":"
            write(iua,format1)spl%title(0),(spl%title(iqty),
     $           spl%title(iqty),iqty=offset,spl%nqty,stride)
            do ix=0,spl%mx
               call cspline_eval(spl,spl%xs(ix),0)
               write(iua,format2)ix,spl%xs(ix),
     $              spl%f(offset:spl%nqty:stride)
            enddo
            write(iua,format1)spl%title(0),(spl%title(iqty),
     $           spl%title(iqty),iqty=offset,spl%nqty,stride)
         endif
c-----------------------------------------------------------------------
c     compute logs.
c-----------------------------------------------------------------------
         if(bin)then
            do ix=0,spl%mx
               xlog=LOG10(ABS(spl%xs(ix)-xend))
               call cspline_eval(spl,spl%xs(ix),0)
               WHERE(spl%f(offset:spl%nqty:stride) /= 0)
                  flog=LOG(spl%f(offset:spl%nqty:stride))
               elseWHERE
                  flog=epsilon
               endWHERE
c-----------------------------------------------------------------------
c     print binary table of node values.
c-----------------------------------------------------------------------
               write(iub)real(spl%xs(ix),4),real(xlog,4),
     $              (real(real(flog(iqty))/alog10,4),
     $              real(AIMAG(flog(iqty))*rtod,4),
     $              iqty=1,SIZE(flog))
            enddo
            write(iub)
         endif
c-----------------------------------------------------------------------
c     print header for interpolated values.
c-----------------------------------------------------------------------
         if(interp)then
            if(out)then
               write(iua,'(/1x,a,i3)')
     $              "interpolated values for offset = ",offset,":"
               write(iua,format1)spl%title(0),
     $              (spl%title(iqty),spl%title(iqty),
     $              iqty=offset,spl%nqty,stride)
            endif
c-----------------------------------------------------------------------
c     print interpolated values.
c-----------------------------------------------------------------------
            do ix=0,spl%mx-1
               dx=(spl%xs(ix+1)-spl%xs(ix))/4
               do j=0,3
                  x=spl%xs(ix)+j*dx
                  xlog=LOG10(ABS(x-xend))
                  call cspline_eval(spl,x,0)
                  if(out)write(iua,format2)ix,x,(spl%f(iqty),
     $                 iqty=offset,spl%nqty,stride)
                  if(bin)then
                     WHERE(spl%f(offset:spl%nqty:stride) /= 0)
                        flog=LOG(spl%f(offset:spl%nqty:stride))
                     elseWHERE
                        flog=epsilon
                     endWHERE
                     write(iub)real(x,4),real(xlog,4),
     $                    (real(Dreal(flog(iqty))/alog10,4),
     $                    real(AIMAG(flog(iqty))*rtod,4),
     $                    iqty=1,SIZE(flog))
                  endif
               enddo
               if(out)write(iua,'(1x)')
            enddo
c-----------------------------------------------------------------------
c     print final interpolated values.
c-----------------------------------------------------------------------
            x=spl%xs(spl%mx)
            xlog=LOG10(ABS(x-xend))
            call cspline_eval(spl,x,0)
            if(out)then
               write(iua,format2)ix,x,(spl%f(iqty),
     $              iqty=offset,spl%nqty,stride)
               write(iua,format1)spl%title(0),(spl%title(iqty),
     $              spl%title(iqty),iqty=offset,spl%nqty,stride)
            endif
            if(bin)then
               WHERE(spl%f(offset:spl%nqty:stride) /= 0)
                  flog=LOG(spl%f(offset:spl%nqty:stride))
               elseWHERE
                  flog=epsilon
               endWHERE
               write(iub)real(x,4),real(xlog,4),
     $              (real(Dreal(flog(iqty))/alog10,4),
     $              real(AIMAG(flog(iqty))*rtod,4),
     $              iqty=1,SIZE(flog))
               write(iub)
            endif
         endif
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_write_log
c-----------------------------------------------------------------------
c     subprogram 12. cspline_int.
c     integrates complex cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_int(spl)

      type(cspline_type), intent(inout) :: spl

      integer :: ix,iqty,ig
      real(r8), dimension(spl%mx) :: dx
      complex(r8), dimension(spl%mx,spl%nqty) :: term,f,f1,f2,f3

      integer, PARAMETER :: mg=4
      real(r8), dimension(mg) :: xg=(1+(/-0.861136311594053_r8,
     $     -0.339981043584856_r8,0.339981043584856_r8,
     $     0.861136311594053_r8/))/2
      real(r8), dimension(mg) :: wg=(/0.347854845137454_r8,
     $     0.652145154862546_r8,0.652145154862546_r8,
     $     0.347854845137454_r8/)/2
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      if(.not.ASSOCIATED(spl%fsi))allocate(spl%fsi(0:spl%mx,spl%nqty))
      dx=spl%xs(1:spl%mx)-spl%xs(0:spl%mx-1)
      term=0
c-----------------------------------------------------------------------
c     compute integrals over intervals.
c-----------------------------------------------------------------------
      do iqty=1,spl%nqty
         if(spl%xpower(1,iqty) == 0 .and. spl%xpower(2,iqty) == 0)then
            term(:,iqty)=dx/12
     $           *(6*(spl%fs(0:spl%mx-1,iqty)+spl%fs(1:spl%mx,iqty))
     $           +dx*(spl%fs1(0:spl%mx-1,iqty)-spl%fs1(1:spl%mx,iqty)))
         else
            do ig=1,mg
               call cspline_all_eval(spl,xg(ig),f,f1,f2,f3,0)
               term(:,iqty)=term(:,iqty)+dx*wg(ig)*f(:,iqty)
            enddo
         endif
      enddo
c-----------------------------------------------------------------------
c     accumulate over intervals.
c-----------------------------------------------------------------------
      spl%fsi(0,:)=0
      do ix=1,spl%mx
         spl%fsi(ix,:)=spl%fsi(ix-1,:)+term(ix,:)
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_int
c-----------------------------------------------------------------------
c     subprogram 13. cspline_triluf.
c     performs tridiagonal LU factorization.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_triluf(a)

      real(r8), dimension(-1:,:), intent(inout) :: a

      integer :: i,j,k,jmin,jmax,n
c-----------------------------------------------------------------------
c     begin loop over rows and define limits.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      do i=1,n
         jmin=MAX(1-i,-1)
         jmax=Min(n-i,1)
c-----------------------------------------------------------------------
c     compute lower elements.
c-----------------------------------------------------------------------
         do j=jmin,-1
            do k=MAX(jmin,j-1),j-1
               a(j,i)=a(j,i)-a(k,i)*a(j-k,i+k)
            enddo
            a(j,i)=a(j,i)*a(0,i+j)
         enddo
c-----------------------------------------------------------------------
c     compute diagonal element
c-----------------------------------------------------------------------
         do k=MAX(jmin,-1),-1
            a(0,i)=a(0,i)-a(k,i)*a(-k,i+k)
         enddo
         a(0,i)=1/a(0,i)
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_triluf
c-----------------------------------------------------------------------
c     subprogram 14. cspline_trilus.
c     performs tridiagonal LU solution.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_trilus(a,x)

      real(r8), dimension(-1:,:), intent(in) :: a
      complex(r8), dimension(:,:), intent(inout) :: x

      integer :: i,j,n
c-----------------------------------------------------------------------
c     down sweep.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      do i=1,n
         do j=MAX(1-i,-1),-1
            x(i,:)=x(i,:)-a(j,i)*x(i+j,:)
         enddo
      enddo
c-----------------------------------------------------------------------
c     up sweep.
c-----------------------------------------------------------------------
      do i=n,1,-1
         do j=1,Min(n-i,1)
            x(i,:)=x(i,:)-a(j,i)*x(i+j,:)
         enddo
         x(i,:)=x(i,:)*a(0,i)
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_trilus
c-----------------------------------------------------------------------
c     subprogram 15. cspline_sherman.
c     uses Sherman-Morrison formula to factor periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_sherman(a)

      real(r8), dimension(-1:,:), intent(inout) :: a

      integer :: j,n
      complex(r8), dimension(SIZE(a,2),1) :: u
c-----------------------------------------------------------------------
c     prepare matrices.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      a(0,1)=a(0,1)-a(-1,1)
      a(0,n)=a(0,n)-a(-1,1)
      u=RESHAPE((/(1.0_r8, 0.0_r8),((0.0_r8, 0.0_r8),j=2,n-1)
     $      ,(1.0_r8, 0.0_r8)/),SHAPE(u))
      call cspline_triluf(a)
      call cspline_trilus(a,u)
      a(-1,1)=real(a(-1,1)/(1+a(-1,1)*(u(1,1)+u(n,1))),r8)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_sherman
c-----------------------------------------------------------------------
c     subprogram 16. cspline_morrison.
c     uses Sherman-Morrison formula to solve periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_morrison(a,x)

      real(r8), dimension(-1:,:), intent(in) :: a
      complex(r8), dimension(:,:), intent(inout) :: x

      integer :: n
      complex(r8), dimension(SIZE(x,1),SIZE(x,2)) :: y
c-----------------------------------------------------------------------
c     solve for x.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      y=x
      call cspline_trilus(a,y)
      x(1,:)=x(1,:)-a(-1,1)*(y(1,:)+y(n,:))
      x(n,:)=x(n,:)-a(-1,1)*(y(1,:)+y(n,:))
      call cspline_trilus(a,x)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_morrison
c-----------------------------------------------------------------------
c     subprogram 17. cspline_thomas.
c     thomas method to solve tri-diagnol (complex) matrix
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_thomas(l,d,u,b,n,m)

      integer, intent(in):: n,m
      real(r8), dimension(n), intent(inout):: d
      real(r8), dimension(n-1), intent(inout):: l,u
      complex(r8), dimension(n,m), intent(inout):: b
      integer:: i
c-----------------------------------------------------------------------
c     calculate tri-diagno matrix
c     l=[A(1,2),A(2,3),...,A(n-1,n)];
c     d=[A(1,1),A(2,2),...,A(n,n)];
c     u=[A(2,1),A(3,2),...,A(n,n-1)];
c     b is n row m column matrix
c-----------------------------------------------------------------------
      do i = 2, n
         l(i-1) = l(i-1)/d(i-1)
         d(i) = d(i) - u(i-1) * l(i-1)
         b(i,:) = b(i,:) - b(i-1,:) * l(i-1)
      enddo

      b(n,:) = b(n,:) / d(n);
      do i = n-1, 1, -1
         b(i,:) = (b(i,:) - u(i) * b(i+1,:)) / d(i);
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_thomas
c-----------------------------------------------------------------------
c     subprogram 18. cspline_get_yp.
c     get yi' with four points for spline boundary condtion.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_get_yp(x,y,xi,yip,nqty)
      integer, intent(in) :: nqty
      integer :: n,nrhs,lda,info,ldb,i
      integer, dimension(4) :: ipiv
      real(r8) :: dx
      real(r8), intent(in) :: xi
      complex(r8), dimension(nqty),intent(out) :: yip
      real(r8), dimension(4), intent(in) :: x
      complex(r8), dimension(4,nqty), intent(in) :: y
      complex(r8), dimension(4,nqty) :: b
      complex(r8), dimension(4,4) :: a
      n=4
      nrhs=nqty
      lda=N
      ldb=N
      a=0
      b=0

      a(1,4)=1
      b(1,:)=y(1,:)
      do i=2,n
         dx=x(i)-x(1)
         a(i,1)=dx*dx*dx
         a(i,2)=dx*dx
         a(i,3)=dx
         a(i,4)=1
         b(i,:)=y(i,:)
      enddo
      call zgesv(n,nrhs,a,lda,ipiv,b,ldb,info)
      dx=xi-x(1)
      yip=(3*b(1,:)*dx+2*b(2,:))*dx+b(3,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_get_yp
c-----------------------------------------------------------------------
c     subprogram 14. cspline_copy.
c     copies one cspline_type to another.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine cspline_copy(spl1,spl2)

      type(cspline_type), intent(in) :: spl1
      type(cspline_type), intent(inout) :: spl2
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      if(ASSOCIATED(spl2%xs))call cspline_dealloc(spl2)
      call cspline_alloc(spl2,spl1%mx,spl1%nqty)
      spl2%xs=spl1%xs
      spl2%fs=spl1%fs
      spl2%fs1=spl1%fs1
      spl2%name=spl1%name
      spl2%title=spl1%title
      spl2%periodic=spl1%periodic
      spl2%xpower=spl1%xpower
      spl2%x0=spl1%x0
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_copy
      end module cspline_mod
