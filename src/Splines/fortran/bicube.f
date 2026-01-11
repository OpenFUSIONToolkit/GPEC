c-----------------------------------------------------------------------
c     file bicube.f.
c     fits functions to bicubic splines.
c     Reference: H. Spaeth, "Spline Algorithms for Curves and Surfaces,"
c     Translated from the German by W. D. Hoskins and H. W. Sager.
c     Utilitas Mathematica Publishing Inc., Winnepeg, 1974.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c      0. bicube_type definition.
c      1. bicube_alloc.
c      2. bicube_dealloc.
c      3. bicube_fit.
c      4. bicube_lsfit.
c      5. bicube_eval.
c      5a. bicube_eval_external.
c      6. bicube_getco.
c      6a. bicube_getco_external.
c      7. bicube_all_eval.
c      8. bicube_all_getco.
c      9. bicube_write_xy.
c     10. bicube_write_yx.
c     11. bicube_write_arrays.
c     12. bicube_copy.
c     13. bicube_extrema.
c-----------------------------------------------------------------------
c     subprogram 0. bicube_type definition.
c     defines bicube_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      module bicube_mod
      use spline_mod
      implicit none

      type :: bicube_type
      integer :: mx,my,nqty,ix,iy
      real(r8), dimension(:,:), allocatable :: xext,yext,fext
      real(r8), dimension(2) :: x0,y0
      real(r8), dimension(:), allocatable :: xs,ys
      real(r8), dimension(:,:), allocatable :: xpower,ypower
      real(r8), dimension(:,:,:), allocatable :: fs,fsx,fsy,fsxy
      real(r8), dimension(:), allocatable :: f,fx,fy,fxx,fxy,fyy
      real(r8), dimension(:,:,:,:,:), allocatable :: cmats
      real(r8), dimension(:,:,:,:,:), allocatable :: gs,gsx,gsy,gsxy,
     $     gsxx,gsyy
      character(6) :: xtitle,ytitle
      character(6), dimension(:), allocatable :: title
      character(6) :: name
      logical, dimension(2) :: periodic
      end type bicube_type

      contains
c-----------------------------------------------------------------------
c     subprogram 1. bicube_alloc.
c     allocates space for bicube_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_alloc(bcs,mx,my,nqty)

      integer, intent(in) :: mx,my,nqty
      type(bicube_type), intent(out) :: bcs
c-----------------------------------------------------------------------
c     set scalars.
c-----------------------------------------------------------------------
      bcs%mx=mx
      bcs%my=my
      bcs%ix=0
      bcs%iy=0
      bcs%nqty=nqty
      bcs%periodic=.false.
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      allocate(bcs%xs(0:mx))
      allocate(bcs%ys(0:my))
      allocate(bcs%fs(0:mx,0:my,nqty))
      allocate(bcs%fsx(0:mx,0:my,nqty))
      allocate(bcs%fsy(0:mx,0:my,nqty))
      allocate(bcs%fsxy(0:mx,0:my,nqty))
      allocate(bcs%title(nqty))
      allocate(bcs%f(nqty))
      allocate(bcs%fx(nqty))
      allocate(bcs%fy(nqty))
      allocate(bcs%fxx(nqty))
      allocate(bcs%fxy(nqty))
      allocate(bcs%fyy(nqty))
      allocate(bcs%xpower(2,nqty),bcs%ypower(2,nqty))
      allocate(bcs%xext(2,nqty),bcs%yext(2,nqty),bcs%fext(2,nqty))
      bcs%xpower=0
      bcs%ypower=0
      bcs%x0=0
      bcs%y0=0
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_alloc
c-----------------------------------------------------------------------
c     subprogram 2. bicube_dealloc.
c     deallocates space for bicube_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_dealloc(bcs)

      type(bicube_type), intent(inout) :: bcs
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      deallocate(bcs%xs)
      deallocate(bcs%ys)
      deallocate(bcs%fs)
      deallocate(bcs%fsx)
      deallocate(bcs%fsy)
      deallocate(bcs%fsxy)
      deallocate(bcs%title)
      deallocate(bcs%f)
      deallocate(bcs%fx)
      deallocate(bcs%fy)
      deallocate(bcs%fxx)
      deallocate(bcs%fxy)
      deallocate(bcs%fyy)
      deallocate(bcs%xpower,bcs%ypower)
      deallocate(bcs%xext,bcs%yext,bcs%fext)
      if(allocated(bcs%cmats))deallocate(bcs%cmats)
      if(allocated(bcs%gs))deallocate(bcs%gs)
      if(allocated(bcs%gsx))deallocate(bcs%gsx)
      if(allocated(bcs%gsy))deallocate(bcs%gsy)
      if(allocated(bcs%gsxx))deallocate(bcs%gsxx)
      if(allocated(bcs%gsxy))deallocate(bcs%gsxy)
      if(allocated(bcs%gsyy))deallocate(bcs%gsyy)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_dealloc
c-----------------------------------------------------------------------
c     subprogram 3. bicube_fit.
c     fits functions to bicubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_fit(bcs,endmode1,endmode2)

      type(bicube_type), intent(inout), TARGET :: bcs
      integer, intent(in) :: endmode1,endmode2

      integer :: iqty,iside,ix,iy
      real(r8), dimension(0:bcs%mx) :: xfac
      real(r8), dimension(0:bcs%my) :: yfac
      type(spline_type) :: spl

      real(r8), dimension(:,:,:), POinTER :: fs,fsx,fsy,fsxy
c-----------------------------------------------------------------------
c     set pointers.
c-----------------------------------------------------------------------
      fs => bcs%fs
      fsx => bcs%fsx
      fsy => bcs%fsy
      fsxy => bcs%fsxy
c-----------------------------------------------------------------------
c     extract x powers.
c-----------------------------------------------------------------------
      do iside=1,2
         do iqty=1,bcs%nqty
            if(bcs%xpower(iside,iqty) /= 0)then
               xfac=1/ABS(bcs%xs-bcs%x0(iside))**bcs%xpower(iside,iqty)
               do iy=0,bcs%my
                  bcs%fs(:,iy,iqty)=bcs%fs(:,iy,iqty)*xfac
               enddo
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     extract y powers.
c-----------------------------------------------------------------------
      do iside=1,2
         do iqty=1,bcs%nqty
            if(bcs%ypower(iside,iqty) /= 0)then
               yfac=1/ABS(bcs%ys-bcs%y0(iside))**bcs%ypower(iside,iqty)
               do ix=0,bcs%mx
                  bcs%fs(ix,:,iqty)=bcs%fs(ix,:,iqty)*yfac
               enddo
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     set periodicity.
c-----------------------------------------------------------------------
      bcs%periodic=(/endmode1 == 2,endmode2 == 2/) !2=periodic
      if(bcs%periodic(1))bcs%fs(bcs%mx,:,:)=bcs%fs(0,:,:)
      if(bcs%periodic(2))bcs%fs(:,bcs%my,:)=bcs%fs(:,0,:)
c-----------------------------------------------------------------------
c     evaluate y derivatives.
c-----------------------------------------------------------------------
      call spline_alloc(spl,bcs%my,bcs%mx+1)
      spl%xs=bcs%ys
      do iqty=1,bcs%nqty
         spl%fs=TRANSPOSE(bcs%fs(:,:,iqty))
         call spline_fit(spl,endmode2)
         bcs%fsy(:,:,iqty)=TRANSPOSE(spl%fs1)
      enddo
      call spline_dealloc(spl)
c-----------------------------------------------------------------------
c     evaluate x derivatives.
c-----------------------------------------------------------------------
      spl%mx=bcs%mx
      spl%nqty=bcs%my+1
      call spline_alloc(spl,bcs%mx,bcs%my+1)
      spl%xs=bcs%xs
      do iqty=1,bcs%nqty
         spl%fs=bcs%fs(:,:,iqty)
         call spline_fit(spl,endmode1)
         bcs%fsx(:,:,iqty)=spl%fs1
      enddo
c-----------------------------------------------------------------------
c     evaluate mixed derivatives.
c-----------------------------------------------------------------------
      do iqty=1,bcs%nqty
         spl%fs=bcs%fsy(:,:,iqty)
         call spline_fit(spl,endmode1)
         bcs%fsxy(:,:,iqty)=spl%fs1
      enddo
      call spline_dealloc(spl)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_fit
c-----------------------------------------------------------------------
c     subprogram 4. bicube_lsfit.
c     least-square fit to cubic splines of piecewise-constant functions.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_lsfit(bcs)

      type(bicube_type), intent(inout) :: bcs

      logical, PARAMETER :: diagnose=.false.
      integer :: ix,iy,nx,ny,mx,my,nrhs,n,info,i,j,k,l,kd,ldab
      integer, dimension(4*(bcs%mx+1)*(bcs%my+1)) :: ipiv
      real(r8), PARAMETER ::
     $     a0=13/35._r8,a1=9/70._r8,b0=11/210._r8,b1=13/420._r8
      real(r8), dimension(0:bcs%mx+1) :: dx
      real(r8), dimension(0:bcs%my+1) :: dy
      real(r8), dimension(4,0:bcs%mx,0:bcs%my,bcs%nqty) :: rhs
      real(r8), dimension(0:bcs%mx+1,0:bcs%my+1,bcs%nqty) :: g
      real(r8), dimension(4,4,-1:1,-1:1,0:bcs%mx,0:bcs%my) :: amat
      real(r8), dimension(:,:), allocatable :: ab
c-----------------------------------------------------------------------
c     define sizes.
c-----------------------------------------------------------------------
      nx=bcs%mx
      ny=bcs%my
      nrhs=bcs%nqty
      kd=4*(nx+1)+7
      ldab=3*kd+1
      n=4*(nx+1)*(ny+1)
c-----------------------------------------------------------------------
c     initialize arrays.
c-----------------------------------------------------------------------
      amat=0
      rhs=0
      dx=0
      dy=0
      g=0
      dx(1:nx)=bcs%xs(1:nx)-bcs%xs(0:nx-1)
      dy(1:ny)=bcs%ys(1:ny)-bcs%ys(0:ny-1)
      g(1:nx,1:ny,:)=bcs%fs(1:nx,1:ny,:)
c-----------------------------------------------------------------------
c     least squares fit, function values.
c-----------------------------------------------------------------------
      do iy=0,ny
         amat(1,1,0,0,0:nx,iy)=a0**2
     $        *(dx(0:nx)+dx(1:nx+1))*(dy(iy)+dy(iy+1))
         amat(1,1,-1,0,0:nx,iy)=a0*a1*dx(0:nx)*(dy(iy)+dy(iy+1))
         amat(1,1,+1,0,0:nx,iy)=a0*a1*dx(1:nx+1)*(dy(iy)+dy(iy+1))
         amat(1,1,0,-1,0:nx,iy)=a0*a1*(dx(0:nx)+dx(1:nx+1))*dy(iy)
         amat(1,1,0,+1,0:nx,iy)=a0*a1*(dx(0:nx)+dx(1:nx+1))*dy(iy+1)
         amat(1,1,-1,-1,0:nx,iy)=a1**2*dx(0:nx)*dy(iy)
         amat(1,1,-1,+1,0:nx,iy)=a1**2*dx(0:nx)*dy(iy+1)
         amat(1,1,+1,-1,0:nx,iy)=a1**2*dx(1:nx+1)*dy(iy)
         amat(1,1,+1,+1,0:nx,iy)=a1**2*dx(1:nx+1)*dy(iy+1)
      enddo
c-----------------------------------------------------------------------
c     least squares fit, x derivatiives.
c-----------------------------------------------------------------------
      do iy=0,ny
         amat(1,2,0,0,0:nx,iy)=a0*b0
     $        *(dx(1:nx+1)**2-dx(0:nx)**2)*(dy(iy)+dy(iy+1))
         amat(1,2,-1,0,0:nx,iy)=+a0*b1*(dy(iy)+dy(iy+1))*dx(0:nx)**2
         amat(1,2,+1,0,0:nx,iy)=-a0*b1*(dy(iy)+dy(iy+1))*dx(1:nx+1)**2
         amat(1,2,-1,-1,0:nx,iy)=+a1*b1*dx(0:nx)**2*dy(iy)
         amat(1,2,+1,-1,0:nx,iy)=-a1*b1*dx(1:nx+1)**2*dy(iy)
         amat(1,2,-1,+1,0:nx,iy)=+a1*b1*dx(0:nx)**2*dy(iy+1)
         amat(1,2,+1,+1,0:nx,iy)=-a1*b1*dx(1:nx+1)**2*dy(iy+1)
      enddo
c-----------------------------------------------------------------------
c     least squares fit, y derivatiives.
c-----------------------------------------------------------------------
      do ix=0,nx
         amat(1,3,0,0,ix,0:ny)=a0*b0
     $        *(dy(1:ny+1)**2-dy(0:ny)**2)*(dx(ix)+dx(ix+1))
         amat(1,3,-1,0,ix,0:ny)=+a0*b1*(dx(ix)+dx(ix+1))*dy(0:ny)**2
         amat(1,3,+1,0,ix,0:ny)=-a0*b1*(dx(ix)+dx(ix+1))*dy(1:ny+1)**2
         amat(1,3,-1,-1,ix,0:ny)=+a1*b1*dy(0:ny)**2*dx(ix)
         amat(1,3,+1,-1,ix,0:ny)=-a1*b1*dy(1:ny+1)**2*dx(ix)
         amat(1,3,-1,+1,ix,0:ny)=+a1*b1*dy(0:ny)**2*dx(ix+1)
         amat(1,3,+1,+1,ix,0:ny)=-a1*b1*dy(1:ny+1)**2*dx(ix+1)
      enddo
c-----------------------------------------------------------------------
c     least squares fit, mixed derivatives.
c-----------------------------------------------------------------------
      do iy=0,ny
         amat(1,4,0,0,0:nx,iy)=b0**2
     $        *(dx(0:nx)**2-dx(1:nx+1)**2)*(dy(iy)**2-dy(iy+1)**2)
         amat(1,4,-1,0,0:nx,iy)=+b0*b1*dx(0:nx)**2
     $        *(dy(iy)**2+dy(iy+1)**2)
         amat(1,4,+1,0,0:nx,iy)=-b0*b1*dx(1:nx+1)**2
     $        *(dy(iy)**2+dy(iy+1)**2)
         amat(1,4,0,-1,0:nx,iy)=+b0*b1*(dx(0:nx)**2+dx(1:nx+1)**2)**2
     $        *dy(iy)**2
         amat(1,4,0,+1,0:nx,iy)=-b0*b1*(dx(0:nx)**2+dx(1:nx+1)**2)**2
     $        *dy(iy+1)
         amat(1,4,-1,-1,0:nx,iy)=b1**2*dx(0:nx)**2*dy(iy)**2
         amat(1,4,-1,+1,0:nx,iy)=-b1**2*dx(0:nx)**2*dy(iy+1)**2
         amat(1,4,+1,-1,0:nx,iy)=-b1**2*dx(1:nx+1)**2*dy(iy)**2
         amat(1,4,+1,+1,0:nx,iy)=b1**2*dx(1:nx+1)**2*dy(iy+1)**2
      enddo
c-----------------------------------------------------------------------
c     least squares fit, rhs.
c-----------------------------------------------------------------------
      do iy=0,ny
         do ix=0,nx
            rhs(1,ix,iy,:)
     $           =(dx(ix)*g(ix,iy,:)+dx(ix+1)*g(ix+1,iy,:))*dy(iy)
     $           +(dx(ix)*g(ix,iy+1,:)+dx(ix+1)*g(ix+1,iy+1,:))*dy(iy+1)
         enddo
      enddo
      rhs=rhs/4
c-----------------------------------------------------------------------
c     continuity of second x-derivatives.
c-----------------------------------------------------------------------
      do ix=1,nx-1
         amat(2,1,-1,0,ix,0:ny)=3/dx(ix)**2
         amat(2,1,0,0,ix,0:ny)=3/dx(ix+1)**2-3/dx(ix)**2
         amat(2,1,1,0,ix,0:ny)=-3/dx(ix+1)**2
         amat(2,2,-1,0,ix,0:ny)=1/dx(ix)
         amat(2,2,0,0,ix,0:ny)=2/dx(ix)+2/dx(ix+1)
         amat(2,2,1,0,ix,0:ny)=1/dx(ix+1)
      enddo
      amat(2,2,0,0,0:nx:nx,0:ny)=1
c-----------------------------------------------------------------------
c     continuity of second y-derivatives.
c-----------------------------------------------------------------------
      do iy=1,ny-1
         amat(3,1,0,-1,0:nx,iy)=3/dy(iy)**2
         amat(3,1,0,0,0:nx,iy)=3/dy(iy+1)**2-3/dy(iy)**2
         amat(3,1,0,1,0:nx,iy)=-3/dy(iy+1)**2
         amat(3,3,0,-1,0:nx,iy)=1/dy(iy)
         amat(3,3,0,0,0:nx,iy)=2/dy(iy)+2/dy(iy+1)
         amat(3,3,0,1,0:nx,iy)=1/dy(iy+1)
      enddo
      amat(3,3,0,0,0:nx,0:ny:ny)=1
c-----------------------------------------------------------------------
c     continuity of mixed second derivatives.
c-----------------------------------------------------------------------
      do ix=1,nx-1
         amat(4,3,-1,0,ix,0:ny)=3/dx(ix)**2
         amat(4,3,0,0,ix,0:ny)=3/dx(ix+1)**2-3/dx(ix)**2
         amat(4,3,1,0,ix,0:ny)=-3/dx(ix+1)**2
         amat(4,4,-1,0,ix,0:ny)=1/dx(ix)
         amat(4,4,0,0,ix,0:ny)=2/dx(ix)+2/dx(ix+1)
         amat(4,4,1,0,ix,0:ny)=1/dx(ix+1)
      enddo
      amat(4,4,0,0,0:nx:nx,0:ny)=1
c-----------------------------------------------------------------------
c     transfer matrix to lapack band storage.
c-----------------------------------------------------------------------
      allocate(ab(ldab,n))
      ab=0
      do ix=0,nx
         do mx=MAX(-ix,-1),Min(nx-ix,1)
            do iy=0,ny
               do my=MAX(-iy,-1),Min(ny-iy,1)
                  do k=1,4
                     do l=1,4
                        i=4*(iy*(nx+1)+ix)+k
                        j=4*((iy+my)*(nx+1)+ix+mx)+l
                        ab(2*kd+1+i-j,j)=amat(k,l,mx,my,ix,iy)
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo
c-----------------------------------------------------------------------
c     factor and solve.
c-----------------------------------------------------------------------
      call dgbtrf(n,n,kd,kd,ab,ldab,ipiv,info)
      call dgbtrs('N',n,kd,kd,nrhs,ab,ldab,ipiv,rhs,n,info)
      deallocate(ab)
c-----------------------------------------------------------------------
c     compute output.
c-----------------------------------------------------------------------
      bcs%fs=rhs(1,:,:,:)
      bcs%fsx=rhs(2,:,:,:)
      bcs%fsy=rhs(3,:,:,:)
      bcs%fsxy=rhs(4,:,:,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_lsfit
c-----------------------------------------------------------------------
c     subprogram 5. bicube_eval.
c     evaluates bicubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_eval(bcs,x,y,mode)

      type(bicube_type), intent(inout) :: bcs
      real(r8), intent(in) :: x,y
      integer, intent(in) :: mode

      integer :: i,iqty,iside
      real(r8) :: dx,dy,xx,yy,g,gx,gy,gxx,gyy,gxy,xfac,yfac
      real(r8), dimension (4,4,bcs%nqty) :: c
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      if(bcs%mx==0 .OR. bcs%my==0)then
         write(*, *) 'ERROR: Bicubic spline entitled "',bcs%title,'"'
         if(bcs%mx==0)then
            write(*, *) 'has 0 elements specified for x axis '
     $                  //TRIM(bcs%xtitle)
         else
            write(*, *) 'has 0 elements specified for y axis '
     $                  //TRIM(bcs%ytitle)
         endif
         stop
      endif
      bcs%ix=max(bcs%ix,0)
      bcs%ix=min(bcs%ix,bcs%mx-1)
      bcs%iy=max(bcs%iy,0)
      bcs%iy=min(bcs%iy,bcs%my-1)
      xx=x
      yy=y
c-----------------------------------------------------------------------
c     normalize x interval for periodic splines.
c-----------------------------------------------------------------------
      if(bcs%periodic(1))then
         do
            if(xx < bcs%xs(bcs%mx))EXIT
            xx=xx-bcs%xs(bcs%mx)
         enddo
         do
            if(xx >= bcs%xs(0))EXIT
            xx=xx+bcs%xs(bcs%mx)
         enddo
      endif
c-----------------------------------------------------------------------
c     find x interval.
c-----------------------------------------------------------------------
      do
         if(bcs%ix <= 0)EXIT
         if(xx >= bcs%xs(bcs%ix))EXIT
         bcs%ix=bcs%ix-1
      enddo
      do
         if(bcs%ix >= bcs%mx-1)EXIT
         if(xx < bcs%xs(bcs%ix+1))EXIT
         bcs%ix=bcs%ix+1
      enddo
c-----------------------------------------------------------------------
c     normalize y interval for periodic splines.
c-----------------------------------------------------------------------
      if(bcs%periodic(2))then
         do
            if(yy < bcs%ys(bcs%my))EXIT
            yy=yy-bcs%ys(bcs%my)
         enddo
         do
            if(yy >= bcs%ys(0))EXIT
            yy=yy+bcs%ys(bcs%my)
         enddo
      endif
c-----------------------------------------------------------------------
c     find y interval.
c-----------------------------------------------------------------------
      do
         if(bcs%iy <= 0)EXIT
         if(yy >= bcs%ys(bcs%iy))EXIT
         bcs%iy=bcs%iy-1
      enddo
      do
         if(bcs%iy >= bcs%my-1)EXIT
         if(yy < bcs%ys(bcs%iy+1))EXIT
         bcs%iy=bcs%iy+1
      enddo
c-----------------------------------------------------------------------
c     find offsets and compute local coefficients.
c-----------------------------------------------------------------------
      dx=xx-bcs%xs(bcs%ix)
      dy=yy-bcs%ys(bcs%iy)
      if(allocated(bcs%cmats))then
         c=bcs%cmats(:,:,bcs%ix+1,bcs%iy+1,:)
      else
         c=bicube_getco(bcs)
      endif
c-----------------------------------------------------------------------
c     evaluate f.
c-----------------------------------------------------------------------
      bcs%f=0
      do i=4,1,-1
         bcs%f=bcs%f*dx
     $        +((c(i,4,:)*dy
     $        +c(i,3,:))*dy
     $        +c(i,2,:))*dy
     $        +c(i,1,:)
      enddo
c-----------------------------------------------------------------------
c     evaluate first derivatives of f
c-----------------------------------------------------------------------
      if(mode > 0)then
         bcs%fx=0
         bcs%fy=0
         do i=4,1,-1
            bcs%fy=bcs%fy*dx
     $           +(c(i,4,:)*3*dy
     $           +c(i,3,:)*2)*dy
     $           +c(i,2,:)
            bcs%fx=bcs%fx*dy
     $           +(c(4,i,:)*3*dx
     $           +c(3,i,:)*2)*dx
     $           +c(2,i,:)
         enddo
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives of f
c-----------------------------------------------------------------------
      if(mode > 1)then
         bcs%fxx=0
         bcs%fyy=0
         bcs%fxy=0
         do i=4,1,-1
            bcs%fyy=bcs%fyy*dx
     $           +(c(i,4,:)*3*dy
     $           +c(i,3,:))*2
            bcs%fxx=bcs%fxx*dy
     $           +(c(4,i,:)*3*dx
     $           +c(3,i,:))*2
         enddo
         do i=4,2,-1
            bcs%fxy=bcs%fxy*dx
     $           +((c(i,4,:)*3*dy
     $           +c(i,3,:)*2)*dy
     $           +c(i,2,:))*(i-1)
         enddo
      endif
c-----------------------------------------------------------------------
c     restore x powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=x-bcs%x0(iside)
         do iqty=1,bcs%nqty
            if(bcs%xpower(iside,iqty) == 0)cycle
            xfac=ABS(dx)**bcs%xpower(iside,iqty)
            g=bcs%f(iqty)*xfac
            if(mode > 0)then
               gx=(bcs%fx(iqty)+bcs%f(iqty)
     $              *bcs%xpower(iside,iqty)/dx)*xfac
               gy=bcs%fy(iqty)*xfac
            endif
            if(mode > 1)then
               gxx=(bcs%fxx(iqty)+bcs%xpower(iside,iqty)/dx
     $              *(2*bcs%fx(iqty)+(bcs%xpower(iside,iqty)-1)
     $              *bcs%f(iqty)/dx))*xfac
               gxy=(bcs%fxy(iqty)+bcs%fy(iqty)
     $              *bcs%xpower(iside,iqty)/dx)*xfac
               gyy=bcs%fyy(iqty)*xfac
            endif
            bcs%f(iqty)=g
            if(mode > 0)then
               bcs%fx(iqty)=gx
               bcs%fy(iqty)=gy
            endif
            if(mode > 1)then
               bcs%fxx(iqty)=gxx
               bcs%fxy(iqty)=gxy
               bcs%fyy(iqty)=gyy
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     restore y powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dy=y-bcs%y0(iside)
         do iqty=1,bcs%nqty
            if(bcs%ypower(iside,iqty) == 0)cycle
            yfac=ABS(dy)**bcs%ypower(iside,iqty)
            g=bcs%f(iqty)*yfac
            if(mode > 0)then
               gx=bcs%fx(iqty)*yfac
               gy=(bcs%fy(iqty)+bcs%f(iqty)
     $              *bcs%ypower(iside,iqty)/dy)*yfac
            endif
            if(mode > 1)then
               gxx=bcs%fxx(iqty)*yfac
               gxy=(bcs%fxy(iqty)+bcs%fy(iqty)
     $              *bcs%ypower(iside,iqty)/dy)*yfac
               gyy=(bcs%fyy(iqty)+bcs%ypower(iside,iqty)/dy
     $              *(2*bcs%fy(iqty)+(bcs%ypower(iside,iqty)-1)
     $              *bcs%f(iqty)/dy))*yfac
            endif
            bcs%f(iqty)=g
            if(mode > 0)then
               bcs%fx(iqty)=gx
               bcs%fy(iqty)=gy
            endif
            if(mode > 1)then
               bcs%fxx(iqty)=gxx
               bcs%fxy(iqty)=gxy
               bcs%fyy(iqty)=gyy
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_eval
c-----------------------------------------------------------------------
c     subprogram 5a. bicube_eval_external.
c     evaluates bicubic spline function with external arrays (parallel).
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_eval_external(bcs, x, y, mode,
     $     b_f, b_fx, b_fy, b_fxx, b_fxy, b_fyy)

      type(bicube_type), intent(in) :: bcs
      real(r8), intent(in) :: x,y
      integer, intent(in) :: mode

      integer :: i,iqty,iside
      integer :: ix, iy, i_low, i_high, i_mid
      real(r8) :: dx,dy,xx,yy,g,gx,gy,gxx,gxy,gyy,xfac,yfac
      real(r8), dimension (4,4,bcs%nqty) :: c

      real(r8), dimension(:), intent(inout) :: b_f
      real(r8), dimension(:), intent(inout), optional :: b_fx,b_fy
      real(r8), dimension(:), intent(inout), optional :: b_fxx,
     $     b_fxy,b_fyy
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      xx=x
      yy=y
c-----------------------------------------------------------------------
c     normalize x interval for periodic splines.
c-----------------------------------------------------------------------
      if(bcs%periodic(1))then
         do
            if(xx < bcs%xs(bcs%mx))EXIT
            xx=xx-bcs%xs(bcs%mx)
         enddo
         do
            if(xx >= bcs%xs(0))EXIT
            xx=xx+bcs%xs(bcs%mx)
         enddo
      endif
c-----------------------------------------------------------------------
c     find x interval using Binary search
c-----------------------------------------------------------------------
      i_low = 0
      i_high = bcs%mx - 1
      do while (i_low <= i_high)
         i_mid = i_low + (i_high - i_low) / 2
         if (xx < bcs%xs(i_mid)) then
               i_high = i_mid - 1
         else if (xx >= bcs%xs(i_mid + 1)) then
               i_low = i_mid + 1
         else
               ix = i_mid
               exit
         endif
      end do
      if (i_low > i_high) ix = min(max(i_low - 1, 0), bcs%mx - 1)
c-----------------------------------------------------------------------
c     normalize y interval for periodic splines.
c-----------------------------------------------------------------------
      if(bcs%periodic(2))then
         do
            if(yy < bcs%ys(bcs%my))EXIT
            yy=yy-bcs%ys(bcs%my)
         enddo
         do
            if(yy >= bcs%ys(0))EXIT
            yy=yy+bcs%ys(bcs%my)
         enddo
      endif
c-----------------------------------------------------------------------
c     find y interval.
c-----------------------------------------------------------------------
      i_low = 0
      i_high = bcs%my - 1
      do while (i_low <= i_high)
         i_mid = i_low + (i_high - i_low) / 2
         if (yy < bcs%ys(i_mid)) then
               i_high = i_mid - 1
         else if (yy >= bcs%ys(i_mid + 1)) then
               i_low = i_mid + 1
         else
               iy = i_mid
               exit
         endif
      end do
      if (i_low > i_high) iy = min(max(i_low - 1, 0), bcs%my - 1)
c-----------------------------------------------------------------------
c     find offsets and compute local coefficients.
c-----------------------------------------------------------------------
      dx=xx-bcs%xs(ix)
      dy=yy-bcs%ys(iy)
      call bicube_getco_external_sub(bcs,ix,iy,c)
c-----------------------------------------------------------------------
c     evaluate f.
c-----------------------------------------------------------------------
      b_f=0
      do i=4,1,-1
         b_f=b_f*dx
     $        +((c(i,4,:)*dy
     $        +c(i,3,:))*dy
     $        +c(i,2,:))*dy
     $        +c(i,1,:)
      enddo
c-----------------------------------------------------------------------
c     evaluate first derivatives of f
c-----------------------------------------------------------------------
      if(mode > 0)then
         b_fx=0
         b_fy=0
         do i=4,1,-1
            b_fy=b_fy*dx
     $           +(c(i,4,:)*3*dy
     $           +c(i,3,:)*2)*dy
     $           +c(i,2,:)
            b_fx=b_fx*dy
     $           +(c(4,i,:)*3*dx
     $           +c(3,i,:)*2)*dx
     $           +c(2,i,:)
         enddo
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives of f
c-----------------------------------------------------------------------
      if(mode > 1)then
         b_fxx=0
         b_fyy=0
         b_fxy=0
         do i=4,1,-1
            b_fyy=b_fyy*dx
     $           +(c(i,4,:)*3*dy
     $           +c(i,3,:))*2
            b_fxx=b_fxx*dy
     $           +(c(4,i,:)*3*dx
     $           +c(3,i,:))*2
         enddo
         do i=4,2,-1
            b_fxy=b_fxy*dx
     $           +((c(i,4,:)*3*dy
     $           +c(i,3,:)*2)*dy
     $           +c(i,2,:))*(i-1)
         enddo
      endif
c-----------------------------------------------------------------------
c     restore x powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=x-bcs%x0(iside)
         do iqty=1,bcs%nqty
            if(bcs%xpower(iside,iqty) == 0)cycle
            xfac=ABS(dx)**bcs%xpower(iside,iqty)
            g=b_f(iqty)*xfac
            if(mode > 0)then
               gx=(b_fx(iqty)+b_f(iqty)
     $              *bcs%xpower(iside,iqty)/dx)*xfac
               gy=b_fy(iqty)*xfac
            endif
            if(mode > 1)then
               gxx=(b_fxx(iqty)+bcs%xpower(iside,iqty)/dx
     $              *(2*bcs%fx(iqty)+(bcs%xpower(iside,iqty)-1)
     $              *b_f(iqty)/dx))*xfac
               gxy=(b_fxy(iqty)+bcs%fy(iqty)
     $              *bcs%xpower(iside,iqty)/dx)*xfac
               gyy=b_fyy(iqty)*xfac
            endif
            b_f(iqty)=g
            if(mode > 0)then
               b_fx(iqty)=gx
               b_fy(iqty)=gy
            endif
            if(mode > 1)then
               b_fxx(iqty)=gxx
               b_fxy(iqty)=gxy
               b_fyy(iqty)=gyy
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     restore y powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dy=y-bcs%y0(iside)
         do iqty=1,bcs%nqty
            if(bcs%ypower(iside,iqty) == 0)cycle
            yfac=ABS(dy)**bcs%ypower(iside,iqty)
            g=b_f(iqty)*yfac
            if(mode > 0)then
               gx=b_fx(iqty)*yfac
               gy=(b_fy(iqty)+b_f(iqty)
     $              *bcs%ypower(iside,iqty)/dy)*yfac
            endif
            if(mode > 1)then
               gxx=b_fxx(iqty)*yfac
               gxy=(b_fxy(iqty)+b_fy(iqty)
     $              *bcs%ypower(iside,iqty)/dy)*yfac
               gyy=(b_fyy(iqty)+bcs%ypower(iside,iqty)/dy
     $              *(2*bcs%fy(iqty)+(bcs%ypower(iside,iqty)-1)
     $              *b_f(iqty)/dy))*yfac
            endif
            b_f(iqty)=g
            if(mode > 0)then
               b_fx(iqty)=gx
               b_fy(iqty)=gy
            endif
            if(mode > 1)then
               b_fxx(iqty)=gxx
               b_fxy(iqty)=gxy
               b_fyy(iqty)=gyy
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_eval_external
c-----------------------------------------------------------------------
c     subprogram 6. bicube_getco.
c     computes coefficient matrices.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      FUNCTION bicube_getco(bcs) RESULT(cmat)

      type(bicube_type), intent(in) :: bcs
      real(r8), dimension(4,4,bcs%nqty) :: cmat

      real(r8) :: hxfac,hxfac2,hxfac3
      real(r8) :: hyfac,hyfac2,hyfac3
      real(r8), dimension(3:4,4) :: gxmat,gymat
      real(r8), dimension(4,4,bcs%nqty) :: temp
c-----------------------------------------------------------------------
c     compute gxmat.
c-----------------------------------------------------------------------
      hxfac=1/(bcs%xs(bcs%ix+1)-bcs%xs(bcs%ix))
      hxfac2=hxfac*hxfac
      hxfac3=hxfac2*hxfac
      gxmat(3,1)=-3*hxfac2
      gxmat(3,2)=-2*hxfac
      gxmat(3,3)=3*hxfac2
      gxmat(3,4)=-hxfac
      gxmat(4,1)=2*hxfac3
      gxmat(4,2)=hxfac2
      gxmat(4,3)=-2*hxfac3
      gxmat(4,4)=hxfac2
c-----------------------------------------------------------------------
c     compute gymat.
c-----------------------------------------------------------------------
      hyfac=1/(bcs%ys(bcs%iy+1)-bcs%ys(bcs%iy))
      hyfac2=hyfac*hyfac
      hyfac3=hyfac2*hyfac
      gymat(3,1)=-3*hyfac2
      gymat(3,2)=-2*hyfac
      gymat(3,3)=3*hyfac2
      gymat(3,4)=-hyfac
      gymat(4,1)=2*hyfac3
      gymat(4,2)=hyfac2
      gymat(4,3)=-2*hyfac3
      gymat(4,4)=hyfac2
c-----------------------------------------------------------------------
c     compute smat.
c-----------------------------------------------------------------------
      cmat(1,1,:)=bcs%fs(bcs%ix,bcs%iy,:)
      cmat(1,2,:)=bcs%fsy(bcs%ix,bcs%iy,:)
      cmat(1,3,:)=bcs%fs(bcs%ix,bcs%iy+1,:)
      cmat(1,4,:)=bcs%fsy(bcs%ix,bcs%iy+1,:)
      cmat(2,1,:)=bcs%fsx(bcs%ix,bcs%iy,:)
      cmat(2,2,:)=bcs%fsxy(bcs%ix,bcs%iy,:)
      cmat(2,3,:)=bcs%fsx(bcs%ix,bcs%iy+1,:)
      cmat(2,4,:)=bcs%fsxy(bcs%ix,bcs%iy+1,:)
      cmat(3,1,:)=bcs%fs(bcs%ix+1,bcs%iy,:)
      cmat(3,2,:)=bcs%fsy(bcs%ix+1,bcs%iy,:)
      cmat(3,3,:)=bcs%fs(bcs%ix+1,bcs%iy+1,:)
      cmat(3,4,:)=bcs%fsy(bcs%ix+1,bcs%iy+1,:)
      cmat(4,1,:)=bcs%fsx(bcs%ix+1,bcs%iy,:)
      cmat(4,2,:)=bcs%fsxy(bcs%ix+1,bcs%iy,:)
      cmat(4,3,:)=bcs%fsx(bcs%ix+1,bcs%iy+1,:)
      cmat(4,4,:)=bcs%fsxy(bcs%ix+1,bcs%iy+1,:)
c-----------------------------------------------------------------------
c     multiply by gymat^T.
c-----------------------------------------------------------------------
      temp(:,1:2,:)=cmat(:,1:2,:)
      temp(:,3,:)
     $     =cmat(:,1,:)*gymat(3,1)
     $     +cmat(:,2,:)*gymat(3,2)
     $     +cmat(:,3,:)*gymat(3,3)
     $     +cmat(:,4,:)*gymat(3,4)
      temp(:,4,:)
     $     =cmat(:,1,:)*gymat(4,1)
     $     +cmat(:,2,:)*gymat(4,2)
     $     +cmat(:,3,:)*gymat(4,3)
     $     +cmat(:,4,:)*gymat(4,4)
c-----------------------------------------------------------------------
c     multiply by gxmat.
c-----------------------------------------------------------------------
      cmat(1:2,:,:)=temp(1:2,:,:)
      cmat(3,:,:)
     $     =gxmat(3,1)*temp(1,:,:)
     $     +gxmat(3,2)*temp(2,:,:)
     $     +gxmat(3,3)*temp(3,:,:)
     $     +gxmat(3,4)*temp(4,:,:)
      cmat(4,:,:)
     $     =gxmat(4,1)*temp(1,:,:)
     $     +gxmat(4,2)*temp(2,:,:)
     $     +gxmat(4,3)*temp(3,:,:)
     $     +gxmat(4,4)*temp(4,:,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end FUNCTION bicube_getco
c-----------------------------------------------------------------------
c     subprogram 6a. bicube_getco_external.
c     computes coefficient matrices for external arrays.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      FUNCTION bicube_getco_external(bcs,b_ix,b_iy) RESULT(cmat)

      type(bicube_type), intent(in) :: bcs
      integer, intent(in) :: b_ix, b_iy
      real(r8), dimension(4,4,bcs%nqty) :: cmat

      real(r8) :: hxfac,hxfac2,hxfac3
      real(r8) :: hyfac,hyfac2,hyfac3
      real(r8), dimension(3:4,4) :: gxmat,gymat
      real(r8), dimension(4,4,bcs%nqty) :: temp
c-----------------------------------------------------------------------
c     compute gxmat.
c-----------------------------------------------------------------------
      hxfac=1/(bcs%xs(b_ix+1)-bcs%xs(b_ix))
      hxfac2=hxfac*hxfac
      hxfac3=hxfac2*hxfac
      gxmat(3,1)=-3*hxfac2
      gxmat(3,2)=-2*hxfac
      gxmat(3,3)=3*hxfac2
      gxmat(3,4)=-hxfac
      gxmat(4,1)=2*hxfac3
      gxmat(4,2)=hxfac2
      gxmat(4,3)=-2*hxfac3
      gxmat(4,4)=hxfac2
c-----------------------------------------------------------------------
c     compute gymat.
c-----------------------------------------------------------------------
      hyfac=1/(bcs%ys(b_iy+1)-bcs%ys(b_iy))
      hyfac2=hyfac*hyfac
      hyfac3=hyfac2*hyfac
      gymat(3,1)=-3*hyfac2
      gymat(3,2)=-2*hyfac
      gymat(3,3)=3*hyfac2
      gymat(3,4)=-hyfac
      gymat(4,1)=2*hyfac3
      gymat(4,2)=hyfac2
      gymat(4,3)=-2*hyfac3
      gymat(4,4)=hyfac2
c-----------------------------------------------------------------------
c     compute smat.
c-----------------------------------------------------------------------
      cmat(1,1,:)=bcs%fs(b_ix,b_iy,:)
      cmat(1,2,:)=bcs%fsy(b_ix,b_iy,:)
      cmat(1,3,:)=bcs%fs(b_ix,b_iy+1,:)
      cmat(1,4,:)=bcs%fsy(b_ix,b_iy+1,:)
      cmat(2,1,:)=bcs%fsx(b_ix,b_iy,:)
      cmat(2,2,:)=bcs%fsxy(b_ix,b_iy,:)
      cmat(2,3,:)=bcs%fsx(b_ix,b_iy+1,:)
      cmat(2,4,:)=bcs%fsxy(b_ix,b_iy+1,:)
      cmat(3,1,:)=bcs%fs(b_ix+1,b_iy,:)
      cmat(3,2,:)=bcs%fsy(b_ix+1,b_iy,:)
      cmat(3,3,:)=bcs%fs(b_ix+1,b_iy+1,:)
      cmat(3,4,:)=bcs%fsy(b_ix+1,b_iy+1,:)
      cmat(4,1,:)=bcs%fsx(b_ix+1,b_iy,:)
      cmat(4,2,:)=bcs%fsxy(b_ix+1,b_iy,:)
      cmat(4,3,:)=bcs%fsx(b_ix+1,b_iy+1,:)
      cmat(4,4,:)=bcs%fsxy(b_ix+1,b_iy+1,:)
c-----------------------------------------------------------------------
c     multiply by gymat^T.
c-----------------------------------------------------------------------
      temp(:,1:2,:)=cmat(:,1:2,:)
      temp(:,3,:)
     $     =cmat(:,1,:)*gymat(3,1)
     $     +cmat(:,2,:)*gymat(3,2)
     $     +cmat(:,3,:)*gymat(3,3)
     $     +cmat(:,4,:)*gymat(3,4)
      temp(:,4,:)
     $     =cmat(:,1,:)*gymat(4,1)
     $     +cmat(:,2,:)*gymat(4,2)
     $     +cmat(:,3,:)*gymat(4,3)
     $     +cmat(:,4,:)*gymat(4,4)
c-----------------------------------------------------------------------
c     multiply by gxmat.
c-----------------------------------------------------------------------
      cmat(1:2,:,:)=temp(1:2,:,:)
      cmat(3,:,:)
     $     =gxmat(3,1)*temp(1,:,:)
     $     +gxmat(3,2)*temp(2,:,:)
     $     +gxmat(3,3)*temp(3,:,:)
     $     +gxmat(3,4)*temp(4,:,:)
      cmat(4,:,:)
     $     =gxmat(4,1)*temp(1,:,:)
     $     +gxmat(4,2)*temp(2,:,:)
     $     +gxmat(4,3)*temp(3,:,:)
     $     +gxmat(4,4)*temp(4,:,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end FUNCTION bicube_getco_external
c-----------------------------------------------------------------------
c     subprogram 6b. bicube_getco_external_sub.
c     computes coefficient matrices for external arrays.
c     Subroutine version.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_getco_external_sub(bcs,b_ix,b_iy, cmat)

      type(bicube_type), intent(in) :: bcs
      integer, intent(in) :: b_ix, b_iy
      real(r8), dimension(4,4,bcs%nqty), intent(out) :: cmat

      real(r8) :: hxfac,hxfac2,hxfac3
      real(r8) :: hyfac,hyfac2,hyfac3
      real(r8), dimension(3:4,4) :: gxmat,gymat
      real(r8), dimension(4,4,bcs%nqty) :: temp
c-----------------------------------------------------------------------
c     compute gxmat.
c-----------------------------------------------------------------------
      hxfac=1/(bcs%xs(b_ix+1)-bcs%xs(b_ix))
      hxfac2=hxfac*hxfac
      hxfac3=hxfac2*hxfac
      gxmat(3,1)=-3*hxfac2
      gxmat(3,2)=-2*hxfac
      gxmat(3,3)=3*hxfac2
      gxmat(3,4)=-hxfac
      gxmat(4,1)=2*hxfac3
      gxmat(4,2)=hxfac2
      gxmat(4,3)=-2*hxfac3
      gxmat(4,4)=hxfac2
c-----------------------------------------------------------------------
c     compute gymat.
c-----------------------------------------------------------------------
      hyfac=1/(bcs%ys(b_iy+1)-bcs%ys(b_iy))
      hyfac2=hyfac*hyfac
      hyfac3=hyfac2*hyfac
      gymat(3,1)=-3*hyfac2
      gymat(3,2)=-2*hyfac
      gymat(3,3)=3*hyfac2
      gymat(3,4)=-hyfac
      gymat(4,1)=2*hyfac3
      gymat(4,2)=hyfac2
      gymat(4,3)=-2*hyfac3
      gymat(4,4)=hyfac2
c-----------------------------------------------------------------------
c     compute smat.
c-----------------------------------------------------------------------
      cmat(1,1,:)=bcs%fs(b_ix,b_iy,:)
      cmat(1,2,:)=bcs%fsy(b_ix,b_iy,:)
      cmat(1,3,:)=bcs%fs(b_ix,b_iy+1,:)
      cmat(1,4,:)=bcs%fsy(b_ix,b_iy+1,:)
      cmat(2,1,:)=bcs%fsx(b_ix,b_iy,:)
      cmat(2,2,:)=bcs%fsxy(b_ix,b_iy,:)
      cmat(2,3,:)=bcs%fsx(b_ix,b_iy+1,:)
      cmat(2,4,:)=bcs%fsxy(b_ix,b_iy+1,:)
      cmat(3,1,:)=bcs%fs(b_ix+1,b_iy,:)
      cmat(3,2,:)=bcs%fsy(b_ix+1,b_iy,:)
      cmat(3,3,:)=bcs%fs(b_ix+1,b_iy+1,:)
      cmat(3,4,:)=bcs%fsy(b_ix+1,b_iy+1,:)
      cmat(4,1,:)=bcs%fsx(b_ix+1,b_iy,:)
      cmat(4,2,:)=bcs%fsxy(b_ix+1,b_iy,:)
      cmat(4,3,:)=bcs%fsx(b_ix+1,b_iy+1,:)
      cmat(4,4,:)=bcs%fsxy(b_ix+1,b_iy+1,:)
c-----------------------------------------------------------------------
c     multiply by gymat^T.
c-----------------------------------------------------------------------
      temp(:,1:2,:)=cmat(:,1:2,:)
      temp(:,3,:)
     $     =cmat(:,1,:)*gymat(3,1)
     $     +cmat(:,2,:)*gymat(3,2)
     $     +cmat(:,3,:)*gymat(3,3)
     $     +cmat(:,4,:)*gymat(3,4)
        temp(:,4,:)
     $     =cmat(:,1,:)*gymat(4,1)
     $     +cmat(:,2,:)*gymat(4,2)
     $     +cmat(:,3,:)*gymat(4,3)
     $     +cmat(:,4,:)*gymat(4,4)
c-----------------------------------------------------------------------
c     multiply by gxmat.
c-----------------------------------------------------------------------
      cmat(1:2,:,:)=temp(1:2,:,:)
      cmat(3,:,:)
     $     =gxmat(3,1)*temp(1,:,:)
     $     +gxmat(3,2)*temp(2,:,:)
     $     +gxmat(3,3)*temp(3,:,:)
     $     +gxmat(3,4)*temp(4,:,:)
         cmat(4,:,:)
     $     =gxmat(4,1)*temp(1,:,:)
     $     +gxmat(4,2)*temp(2,:,:)
     $     +gxmat(4,3)*temp(3,:,:)
     $     +gxmat(4,4)*temp(4,:,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_getco_external_sub
c-----------------------------------------------------------------------
c     subprogram 7. bicube_all_eval.
c     evaluates bicubic splines in all intervals.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_all_eval(bcs,dx,dy,f,fx,fy,fxx,fyy,fxy,mode)

      type(bicube_type), intent(inout) :: bcs
      real(r8), intent(in) :: dx,dy
      real(r8), intent(out), dimension(bcs%mx,bcs%my,bcs%nqty) ::
     $     f,fx,fy,fxx,fyy,fxy
      integer, intent(in) :: mode

      integer :: i,ix,iy,iqty,iside
      real(r8), dimension(bcs%mx) :: dxv
      real(r8), dimension(bcs%my) :: dyv

      real(R8), dimension(bcs%mx) :: dxx,xfac
      real(R8), dimension(bcs%my) :: dyy,yfac
      real(R8), dimension(bcs%mx,bcs%my) :: g,gx,gy,gxx,gxy,gyy
c-----------------------------------------------------------------------
c     compute local displacements and coefficients.
c-----------------------------------------------------------------------
      dxv=(bcs%xs(1:bcs%mx)-bcs%xs(0:bcs%mx-1))*dx
      dyv=(bcs%ys(1:bcs%my)-bcs%ys(0:bcs%my-1))*dy
      call bicube_all_getco(bcs)
c-----------------------------------------------------------------------
c     evaluate f.
c-----------------------------------------------------------------------
      f=0
      do i=4,1,-1
         if(i /= 4)then
            do ix=1,bcs%mx
               f(ix,:,:)=f(ix,:,:)*dxv(ix)
            enddo
         endif
         do iy=1,bcs%my
            f(:,iy,:)=f(:,iy,:)
     $           +((bcs%cmats(i,4,:,iy,:)*dyv(iy)
     $           +bcs%cmats(i,3,:,iy,:))*dyv(iy)
     $           +bcs%cmats(i,2,:,iy,:))*dy
     $           +bcs%cmats(i,1,:,iy,:)
         enddo
      enddo
c-----------------------------------------------------------------------
c     evaluate fx.
c-----------------------------------------------------------------------
      if(mode > 0)then
         fx=0
         do i=4,1,-1
            if(i /= 4)then
               do iy=1,bcs%my
                  fx(:,iy,:)=fx(:,iy,:)*dyv(iy)
               enddo
            endif
            do ix=1,bcs%mx
               fx(ix,:,:)=fx(ix,:,:)
     $              +(bcs%cmats(4,i,ix,:,:)*3*dxv(ix)
     $              +bcs%cmats(3,i,ix,:,:)*2)*dxv(ix)
     $              +bcs%cmats(2,i,ix,:,:)
            enddo
         enddo
c-----------------------------------------------------------------------
c     evaluate fy.
c-----------------------------------------------------------------------
         fy=0
         do i=4,1,-1
            if(i /= 4)then
               do ix=1,bcs%mx
                  fy(ix,:,:)=fy(ix,:,:)*dxv(ix)
               enddo
            endif
            do iy=1,bcs%my
               fy(:,iy,:)=fy(:,iy,:)
     $              +(bcs%cmats(i,4,:,iy,:)*3*dyv(iy)
     $              +bcs%cmats(i,3,:,iy,:)*2)*dyv(iy)
     $              +bcs%cmats(i,2,:,iy,:)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     evaluate fxx.
c-----------------------------------------------------------------------
      if(mode > 1)then
         fxx=0
         do i=4,1,-1
            if(i /= 4)then
               do iy=1,bcs%my
                  fxx(:,iy,:)=fxx(:,iy,:)*dyv(iy)
               enddo
            endif
            do ix=1,bcs%mx
               fxx(ix,:,:)=fxx(ix,:,:)
     $              +(bcs%cmats(4,i,ix,:,:)*3*dxv(ix)
     $              +bcs%cmats(3,i,ix,:,:))*2
            enddo
         enddo
c-----------------------------------------------------------------------
c     evaluate fyy and fxy
c-----------------------------------------------------------------------
         fyy=0
         do i=4,1,-1
            if(i /= 4)then
               do ix=1,bcs%mx
                  fyy(ix,:,:)=fyy(ix,:,:)*dxv(ix)
                  fxy(ix,:,:)=fxy(ix,:,:)*dxv(ix)
               enddo
            endif
            do iy=1,bcs%my
               fyy(:,iy,:)=fyy(:,iy,:)
     $              +(bcs%cmats(i,4,:,iy,:)*3*dyv(iy)
     $              +bcs%cmats(i,3,:,iy,:))*2
               fxy(:,iy,:)=fxy(:,iy,:)
     $              +((bcs%cmats(i,4,:,iy,:)*3*dyv(iy)
     $              +bcs%cmats(i,3,:,iy,:)*2)*dyv(iy)
     $              +bcs%cmats(i,2,:,iy,:))*(i-1)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     restore x powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dxx=(bcs%xs(0:bcs%mx-1)+dxv(1:bcs%mx))-bcs%x0(iside)
         do iqty=1,bcs%nqty
            if(bcs%xpower(iside,iqty) == 0)cycle
            xfac=dxx**bcs%xpower(iside,iqty)
            do iy=1,bcs%my
               g(:,iy)=f(:,iy,iqty)*xfac
               if(mode > 0)then
                  gx(:,iy)=(fx(:,iy,iqty)+f(:,iy,iqty)
     $                 *bcs%xpower(iside,iqty)/dxx)*xfac
                  gy(:,iy)=fy(:,iy,iqty)*xfac
               endif
               if(mode > 1)then
                  gxy(:,iy)=(fxy(:,iy,iqty)+fy(:,iy,iqty)
     $                 *bcs%xpower(iside,iqty)/dxx)*xfac
                  gxx(:,iy)=(fxx(:,iy,iqty)+bcs%xpower(iside,iqty)/dxx
     $                 *(2*fx(:,iy,iqty)+(bcs%xpower(iside,iqty)-1)
     $                 *f(:,iy,iqty)/dxx))*xfac
                  gyy(:,iy)=fyy(:,iy,iqty)*xfac
               endif
               f(:,iy,iqty)=g(:,iy)
               if(mode > 0)then
                  fx(:,iy,iqty)=gx(:,iy)
                  fy(:,iy,iqty)=gy(:,iy)
               endif
               if(mode > 1)then
                  fxx(:,iy,iqty)=gxx(:,iy)
                  fxy(:,iy,iqty)=gxy(:,iy)
                  fyy(:,iy,iqty)=gyy(:,iy)
               endif
            enddo
         enddo
      enddo
c-----------------------------------------------------------------------
c     restore y powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dyy=(bcs%ys(0:bcs%my-1)+dyv(1:bcs%my))-bcs%y0(iside)
         do iqty=1,bcs%nqty
            if(bcs%ypower(iside,iqty) == 0)cycle
            yfac=dyy**bcs%ypower(iside,iqty)
            do ix=1,bcs%mx
               g(ix,:)=f(ix,:,iqty)*yfac
               if(mode > 0)then
                  gy(ix,:)=(fy(ix,:,iqty)+f(ix,:,iqty)
     $                 *bcs%ypower(iside,iqty)/dyy)*yfac
                  gx(ix,:)=fx(ix,:,iqty)*yfac
               endif
               if(mode > 1)then
                  gxx(ix,:)=fxx(ix,:,iqty)*yfac
                  gxy(ix,:)=(fxy(ix,:,iqty)+fx(ix,:,iqty)
     $                 *bcs%ypower(iside,iqty)/dyy)*yfac
                  gyy(ix,:)=(fyy(ix,:,iqty)+bcs%ypower(iside,iqty)/dyy
     $                 *(2*fy(ix,:,iqty)+(bcs%ypower(iside,iqty)-1)
     $                 *f(ix,:,iqty)/dyy))*yfac
               endif
               f(ix,:,iqty)=g(ix,:)
               if(mode > 0)then
                  fx(ix,:,iqty)=gx(ix,:)
                  fy(ix,:,iqty)=gy(ix,:)
               endif
               if(mode > 1)then
                  fxx(ix,:,iqty)=gxx(ix,:)
                  fxy(ix,:,iqty)=gxy(ix,:)
                  fyy(ix,:,iqty)=gyy(ix,:)
               endif
            enddo
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_all_eval
c-----------------------------------------------------------------------
c     subprogram 8. bicube_all_getco.
c     computes coefficient matrices in all intervals.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_all_getco(bcs)

      type(bicube_type), intent(inout) :: bcs

      integer :: ix,iy
      real(r8), dimension(bcs%mx) :: hxfac,hxfac2,hxfac3
      real(r8), dimension(bcs%my) :: hyfac,hyfac2,hyfac3
      real(r8), dimension(3:4,4,bcs%mx) :: gxmat
      real(r8), dimension(3:4,4,bcs%my) :: gymat
      real(r8), dimension(4,4,bcs%mx,bcs%my,bcs%nqty) :: temp
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      if(allocated(bcs%cmats))then
         return
      else
         allocate(bcs%cmats(4,4,bcs%mx,bcs%my,bcs%nqty))
      endif
c-----------------------------------------------------------------------
c     compute gxmat.
c-----------------------------------------------------------------------
      hxfac=1/(bcs%xs(1:bcs%mx)-bcs%xs(0:bcs%mx-1))
      hxfac2=hxfac*hxfac
      hxfac3=hxfac2*hxfac
      gxmat(3,1,:)=-3*hxfac2
      gxmat(3,2,:)=-2*hxfac
      gxmat(3,3,:)=3*hxfac2
      gxmat(3,4,:)=-hxfac
      gxmat(4,1,:)=2*hxfac3
      gxmat(4,2,:)=hxfac2
      gxmat(4,3,:)=-2*hxfac3
      gxmat(4,4,:)=hxfac2
c-----------------------------------------------------------------------
c     compute gymat.
c-----------------------------------------------------------------------
      hyfac=1/(bcs%ys(1:bcs%my)-bcs%ys(0:bcs%my-1))
      hyfac2=hyfac*hyfac
      hyfac3=hyfac2*hyfac
      gymat(3,1,:)=-3*hyfac2
      gymat(3,2,:)=-2*hyfac
      gymat(3,3,:)=3*hyfac2
      gymat(3,4,:)=-hyfac
      gymat(4,1,:)=2*hyfac3
      gymat(4,2,:)=hyfac2
      gymat(4,3,:)=-2*hyfac3
      gymat(4,4,:)=hyfac2
c-----------------------------------------------------------------------
c     compute smat.
c-----------------------------------------------------------------------
      bcs%cmats(1,1,:,:,:)=bcs%fs(0:bcs%mx-1,0:bcs%my-1,:)
      bcs%cmats(1,2,:,:,:)=bcs%fsy(0:bcs%mx-1,0:bcs%my-1,:)
      bcs%cmats(1,3,:,:,:)=bcs%fs(0:bcs%mx-1,1:bcs%my,:)
      bcs%cmats(1,4,:,:,:)=bcs%fsy(0:bcs%mx-1,1:bcs%my,:)
      bcs%cmats(2,1,:,:,:)=bcs%fsx(0:bcs%mx-1,0:bcs%my-1,:)
      bcs%cmats(2,2,:,:,:)=bcs%fsxy(0:bcs%mx-1,0:bcs%my-1,:)
      bcs%cmats(2,3,:,:,:)=bcs%fsx(0:bcs%mx-1,1:bcs%my,:)
      bcs%cmats(2,4,:,:,:)=bcs%fsxy(0:bcs%mx-1,1:bcs%my,:)
      bcs%cmats(3,1,:,:,:)=bcs%fs(1:bcs%mx,0:bcs%my-1,:)
      bcs%cmats(3,2,:,:,:)=bcs%fsy(1:bcs%mx,0:bcs%my-1,:)
      bcs%cmats(3,3,:,:,:)=bcs%fs(1:bcs%mx,1:bcs%my,:)
      bcs%cmats(3,4,:,:,:)=bcs%fsy(1:bcs%mx,1:bcs%my,:)
      bcs%cmats(4,1,:,:,:)=bcs%fsx(1:bcs%mx,0:bcs%my-1,:)
      bcs%cmats(4,2,:,:,:)=bcs%fsxy(1:bcs%mx,0:bcs%my-1,:)
      bcs%cmats(4,3,:,:,:)=bcs%fsx(1:bcs%mx,1:bcs%my,:)
      bcs%cmats(4,4,:,:,:)=bcs%fsxy(1:bcs%mx,1:bcs%my,:)
c-----------------------------------------------------------------------
c     multiply by gymat^T.
c-----------------------------------------------------------------------
      temp(:,1:2,:,:,:)=bcs%cmats(:,1:2,:,:,:)
      do iy=1,bcs%my
         temp(:,3,:,iy,:)
     $        =bcs%cmats(:,1,:,iy,:)*gymat(3,1,iy)
     $        +bcs%cmats(:,2,:,iy,:)*gymat(3,2,iy)
     $        +bcs%cmats(:,3,:,iy,:)*gymat(3,3,iy)
     $        +bcs%cmats(:,4,:,iy,:)*gymat(3,4,iy)
         temp(:,4,:,iy,:)
     $        =bcs%cmats(:,1,:,iy,:)*gymat(4,1,iy)
     $        +bcs%cmats(:,2,:,iy,:)*gymat(4,2,iy)
     $        +bcs%cmats(:,3,:,iy,:)*gymat(4,3,iy)
     $        +bcs%cmats(:,4,:,iy,:)*gymat(4,4,iy)
      enddo
c-----------------------------------------------------------------------
c     multiply by gxmat.
c-----------------------------------------------------------------------
      bcs%cmats(1:2,:,:,:,:)=temp(1:2,:,:,:,:)
      do ix=1,bcs%mx
         bcs%cmats(3,:,ix,:,:)
     $        =gxmat(3,1,ix)*temp(1,:,ix,:,:)
     $        +gxmat(3,2,ix)*temp(2,:,ix,:,:)
     $        +gxmat(3,3,ix)*temp(3,:,ix,:,:)
     $        +gxmat(3,4,ix)*temp(4,:,ix,:,:)
         bcs%cmats(4,:,ix,:,:)
     $        =gxmat(4,1,ix)*temp(1,:,ix,:,:)
     $        +gxmat(4,2,ix)*temp(2,:,ix,:,:)
     $        +gxmat(4,3,ix)*temp(3,:,ix,:,:)
     $        +gxmat(4,4,ix)*temp(4,:,ix,:,:)
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_all_getco
c-----------------------------------------------------------------------
c     subprogram 9. bicube_write_xy.
c     produces ascii and binary output for bicubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_write_xy(bcs,out,bin,iua,iub,interp)

      type(bicube_type), intent(inout) :: bcs
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      logical, intent(in) :: interp

      integer :: ix,iy,jx,jy,iqty
      real(r8) :: x,y,dx,dy

      character(80) :: format1,format2
c-----------------------------------------------------------------------
c     write formats.
c-----------------------------------------------------------------------
 10   format(1x,"iy = ",i3,", ",a6," = ",1p,e11.3)
 20   format(1x,"iy = ",i3,", jy = ",i1,", ",a6," = ",1p,e11.3)
 30   format('(/3x,"ix",4x,a,1x,',i3.3,'(4x,a6,1x)/)')
 40   format('(i5,1p,e11.3,',i3.3,'e11.3)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not. (out. OR. bin))return
c-----------------------------------------------------------------------
c     create format statements.
c-----------------------------------------------------------------------
      if(out)then
         write(format1,30)bcs%nqty
         write(format2,40)bcs%nqty
      endif
c-----------------------------------------------------------------------
c     write input data.
c-----------------------------------------------------------------------
      if(out)write(iua,'(1x,a/)')"input data"
      do iy=0,bcs%my
         y=bcs%ys(iy)
         if(out)then
            write(iua,10)iy,bcs%ytitle,bcs%ys(iy)
            write(iua,format1)bcs%xtitle,
     $           (bcs%title(iqty),iqty=1,bcs%nqty)
         endif
         do ix=0,bcs%mx
            x=bcs%xs(ix)
            call bicube_eval(bcs,x,y,0)
            if(out)write(iua,format2)ix,x,bcs%f
            if(bin)write(iub)real(x,4),real(bcs%f,4)
         enddo
         if(out)write(iua,format1)bcs%xtitle,
     $        (bcs%title(iqty),iqty=1,bcs%nqty)
         if(bin)write(iub)
      enddo
c-----------------------------------------------------------------------
c     begin loops over y for interpolated data.
c-----------------------------------------------------------------------
      if(interp)then
         if(out)write(iua,'(1x,a/)')"interpolated data"
         do iy=0,bcs%my-1
            dy=(bcs%ys(iy+1)-bcs%ys(iy))/4
            do jy=0,4
               y=bcs%ys(iy)+dy*jy
               if(out)then
                  write(iua,20)iy,jy,bcs%ytitle,y
                  write(iua,format1)bcs%xtitle,
     $                 (bcs%title(iqty),iqty=1,bcs%nqty)
               endif
c-----------------------------------------------------------------------
c     begin loops over x for interpolated data.
c-----------------------------------------------------------------------
               do ix=0,bcs%mx-1
                  dx=(bcs%xs(ix+1)-bcs%xs(ix))/4
                  do jx=0,4
                     x=bcs%xs(ix)+dx*jx
                     call bicube_eval(bcs,x,y,0)
                     if(out)write(iua,format2)ix,x,bcs%f
                     if(bin)write(iub)real(x,4),real(bcs%f,4)
                  enddo
                  if(out)write(iua,'()')
               enddo
c-----------------------------------------------------------------------
c     complete loops over y.
c-----------------------------------------------------------------------
               if(out)write(iua,format1)bcs%xtitle,
     $                 (bcs%title(iqty),iqty=1,bcs%nqty)
               if(bin)write(iub)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_write_xy
c-----------------------------------------------------------------------
c     subprogram 10. bicube_write_yx.
c     produces ascii and binary output for bicubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_write_yx(bcs,out,bin,iua,iub,interp)

      type(bicube_type), intent(inout) :: bcs
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      logical, intent(in) :: interp

      integer :: ix,iy,jx,jy,iqty
      real(r8) :: x,y,dx,dy

      character(80) :: format1,format2
c-----------------------------------------------------------------------
c     write formats.
c-----------------------------------------------------------------------
 10   format(1x,"ix = ",i3,", ",a6," = ",1p,e11.3)
 20   format(1x,"ix = ",i3,", jx = ",i1,", ",a6," = ",1p,e11.3)
 30   format('(/4x,"iy",4x,a6,1x,',i3.3,'(4x,a6,1x)/)')
 40   format('(i6,1p,e11.3,',i3.3,'e11.3)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not. (out. OR. bin))return
c-----------------------------------------------------------------------
c     create format statements.
c-----------------------------------------------------------------------
      if(out)then
         write(format1,30)bcs%nqty
         write(format2,40)bcs%nqty
         write(iua,'(1x,a/)')"input data"
      endif
c-----------------------------------------------------------------------
c     write input data.
c-----------------------------------------------------------------------
      do ix=0,bcs%mx
         x=bcs%xs(ix)
         if(out)then
            write(iua,10)ix,bcs%xtitle,bcs%xs(ix)
            write(iua,format1)bcs%ytitle,
     $           (bcs%title(iqty),iqty=1,bcs%nqty)
         endif
         do iy=0,bcs%my
            y=bcs%ys(iy)
            call bicube_eval(bcs,x,y,0)
            if(out)write(iua,format2)iy,y,bcs%f
            if(bin)write(iub)real(y,4),real(bcs%f,4)
         enddo
         if(out)write(iua,format1)bcs%ytitle,
     $        (bcs%title(iqty),iqty=1,bcs%nqty)
         if(bin)write(iub)
      enddo
c-----------------------------------------------------------------------
c     begin loops over x for interpolated data.
c-----------------------------------------------------------------------
      if(interp)then
         if(out)write(iua,'(1x,a/)')"interpolated data"
         do ix=0,bcs%mx-1
            dx=(bcs%xs(ix+1)-bcs%xs(ix))/4
            do jx=0,4
               x=bcs%xs(ix)+dx*jx
               if(out)then
                  write(iua,20)ix,jx,bcs%xtitle,x
                  write(iua,format1)bcs%ytitle,
     $                 (bcs%title(iqty),iqty=1,bcs%nqty)
               endif
c-----------------------------------------------------------------------
c     begin loops over y for interpolated data.
c-----------------------------------------------------------------------
               do iy=0,bcs%my-1
                  dy=(bcs%ys(iy+1)-bcs%ys(iy))/4
                  do jy=0,4
                     y=bcs%ys(iy)+dy*jy
                     call bicube_eval(bcs,x,y,0)
                     if(out)write(iua,format2)iy,y,bcs%f
                     if(bin)write(iub)real(y,4),real(bcs%f,4)
                  enddo
                  if(out)write(iua,'()')
               enddo
c-----------------------------------------------------------------------
c     complete loops over x.
c-----------------------------------------------------------------------
               if(out)write(iua,format1)bcs%ytitle,
     $                 (bcs%title(iqty),iqty=1,bcs%nqty)
               if(bin)write(iub)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_write_yx
c-----------------------------------------------------------------------
c     subprogram 11. bicube_write_arrays.
c     produces ascii and binary output for bicubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_write_arrays(bcs,out,iua,iqty)

      type(bicube_type), intent(inout) :: bcs
      logical, intent(in) :: out
      integer, intent(in) :: iua,iqty

      character(80) :: format1,format2
      integer :: ix,iy,my
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   format('(/2x,"ix/iy",',i3.3,'(3x,i3.3,5x)/)')
 20   format('(i5,1p,',i3.3,'e11.3)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not. out)return
      my=Min(bcs%my,32)
      write(iua,'(a,i2/)')"iqty = ",iqty
c-----------------------------------------------------------------------
c     write fs.
c-----------------------------------------------------------------------
      write(format1,10)my+1
      write(format2,20)my+1
      write(iua,"(a)")"fs:"
      write(iua,format1)(iy,iy=0,my)
      write(iua,format2)(ix,(bcs%fs(ix,iy,iqty),iy=0,my),
     $     ix=0,bcs%mx)
      write(iua,format1)(iy,iy=0,my)
c-----------------------------------------------------------------------
c     write fsx.
c-----------------------------------------------------------------------
      write(iua,"(a)")"fsx:"
      write(iua,format1)(iy,iy=0,my)
      write(iua,format2)(ix,(bcs%fsx(ix,iy,iqty),iy=0,my),
     $     ix=0,bcs%mx)
      write(iua,format1)(iy,iy=0,my)
c-----------------------------------------------------------------------
c     write fsy.
c-----------------------------------------------------------------------
      write(iua,"(a)")"fsy:"
      write(iua,format1)(iy,iy=0,my)
      write(iua,format2)(ix,(bcs%fsy(ix,iy,iqty),iy=0,my),
     $     ix=0,bcs%mx)
      write(iua,format1)(iy,iy=0,my)
c-----------------------------------------------------------------------
c     write fsxy.
c-----------------------------------------------------------------------
      write(iua,"(a)")"fsxy:"
      write(iua,format1)(iy,iy=0,my)
      write(iua,format2)(ix,(bcs%fsxy(ix,iy,iqty),iy=0,my),
     $     ix=0,bcs%mx)
      write(iua,format1)(iy,iy=0,my)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_write_arrays
c-----------------------------------------------------------------------
c     subprogram 12. bicube_copy.
c     copies one bicube type to another.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_copy(bcs1,bcs2)

      type(bicube_type), intent(in) :: bcs1
      type(bicube_type), intent(inout) :: bcs2
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      if(allocated(bcs2%xs))call bicube_dealloc(bcs2)
      call bicube_alloc(bcs2,bcs1%mx,bcs1%my,bcs1%nqty)
      bcs2%xs=bcs1%xs
      bcs2%ys=bcs1%ys
      bcs2%fs=bcs1%fs
      bcs2%fsx=bcs1%fsx
      bcs2%fsy=bcs1%fsy
      bcs2%fsxy=bcs1%fsxy
      bcs2%name=bcs1%name
      bcs2%xtitle=bcs1%xtitle
      bcs2%ytitle=bcs1%ytitle
      bcs2%title=bcs1%title
      bcs2%periodic=bcs1%periodic
      bcs2%xpower=bcs1%xpower
      bcs2%ypower=bcs1%ypower
      bcs2%x0=bcs1%x0
      bcs2%y0=bcs1%y0
      bcs2%xext=bcs1%xext
      bcs2%yext=bcs1%yext
      bcs2%fext=bcs1%fext
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_copy
c-----------------------------------------------------------------------
c     subprogram 13. bicube_extrema.
c     finds extrema.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bicube_extrema(bcs)

      type(bicube_type), intent(inout) :: bcs

      character(80) :: message
      integer :: iext,iqty,it
      integer, PARAMETER :: itmax=20
      integer, dimension(2,2) :: jext
      real(r8), PARAMETER :: eps=1e-10
      real(r8) :: x,y,dx,dy,lx,ly,lf,f,df,adet
      real(r8), dimension(2,2) :: amat,ainv
c-----------------------------------------------------------------------
c     compute lengths.
c-----------------------------------------------------------------------
      lx=bcs%xs(bcs%mx)-bcs%xs(0)
      ly=bcs%ys(bcs%my)-bcs%ys(0)
c-----------------------------------------------------------------------
c     start loops over iqty.
c-----------------------------------------------------------------------
      do iqty=1,bcs%nqty
         jext(:,1)=MinLOC(bcs%fs(:,:,iqty))-1
         jext(:,2)=MAXLOC(bcs%fs(:,:,iqty))-1
         bcs%fext(1,iqty)=bcs%fs(jext(1,1),jext(2,1),iqty)
         bcs%fext(2,iqty)=bcs%fs(jext(1,2),jext(2,2),iqty)
         lf=bcs%fext(2,iqty)-bcs%fext(1,iqty)
c-----------------------------------------------------------------------
c     start loops over extrema.
c-----------------------------------------------------------------------
         do iext=1,2
            x=bcs%xs(jext(1,iext))
            y=bcs%ys(jext(2,iext))
            f=HUGE(f)
            dx=lx
            dy=ly
            it=0
c-----------------------------------------------------------------------
c     locate extema by newton iteration.
c-----------------------------------------------------------------------
            do
               call bicube_eval(bcs,x,y,2)
               df=bcs%f(iqty)-f
               if(ABS(dx) < eps*lx .OR. ABS(dy) < eps*ly
     $              .OR. ABS(df) < eps*lf .OR. it >= itmax)EXIT
               it=it+1
               f=bcs%f(iqty)
               amat(1,1)=bcs%fxx(iqty)
               amat(2,2)=bcs%fyy(iqty)
               amat(1,2)=bcs%fxy(iqty)
               amat(2,1)=bcs%fxy(iqty)
               adet=amat(1,1)*amat(2,2)-amat(1,2)*amat(2,1)
               ainv(1,1)=amat(2,2)
               ainv(2,2)=amat(1,1)
               ainv(1,2)=-amat(1,2)
               ainv(2,1)=-amat(2,1)
               ainv=ainv/adet
               dx=-ainv(1,1)*bcs%fx(iqty)-ainv(1,2)*bcs%fy(iqty)
               dy=-ainv(2,1)*bcs%fx(iqty)-ainv(2,2)*bcs%fy(iqty)
               x=x+dx
               y=y+dy
            enddo
c-----------------------------------------------------------------------
c     abort on failure.
c-----------------------------------------------------------------------
            if(it >= itmax)then
               write(message,'(a,i3,a)')
     $              "bicube_extrema: convergence failure for iqty = ",
     $              iqty,"."
               call program_stop(message)
            endif
c-----------------------------------------------------------------------
c     finish loops over iext and iqty.
c-----------------------------------------------------------------------
            bcs%xext(iext,iqty)=x
            bcs%yext(iext,iqty)=y
            bcs%fext(iext,iqty)=bcs%f(iqty)
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_extrema
      end module bicube_mod
