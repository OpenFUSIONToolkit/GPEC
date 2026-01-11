c-----------------------------------------------------------------------
c     file spline.f
c     fits functions to cubic splines.
c     Reference: H. Spaeth, "Spline Algorithms for Curves and Surfaces,"
c     Translated from the German by W. D. Hoskins and H. W. Sager.
c     Utilitas Mathematica Publishing Inc., Winnepeg, 1974.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. spline_mod.
c     1. spline_alloc.
c     2. spline_dealloc.
c     3. spline_fit.
c     4. spline_fit_ahg.
c     5. spline_fit_classic.
c     6. spline_fit_ha.
c     7. spline_fac.
c     8. spline_eval.
c     9. spline_eval_external.
c     10. spline_all_eval.
c     11. spline_write1.
c     12. spline_write2.
c     13. spline_int.
c     14. spline_triluf.
c     15. spline_trilus.
c     16. spline_sherman.
c     17. spline_morrison.
c     18. spline_copy.
c     19. spline_fill_matrix.
c     20. spline_get_yp.
c     21. spline_thomas.
c     22. spline_roots.
c     23. spline_refine_root.
c     24. bubble_sort.
c-----------------------------------------------------------------------
c     subprogram 0. spline_type definition.
c     defines spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      module spline_mod
      use defs_mod
      implicit none

      logical :: use_classic_splines = .false.
      type :: spline_type
      integer :: mx,nqty,ix
      real(r8), dimension(:), allocatable :: xs,f,f1,f2,f3
      real(r8), dimension(:,:), allocatable :: fs,fs1,fsi,xpower
      real(r8), dimension(2) :: x0
      character(6), dimension(:), allocatable :: title
      character(6) :: name
      logical :: periodic=.false., allocated=.false.
      end type spline_type

      contains
c-----------------------------------------------------------------------
c     subprogram 1. spline_alloc.
c     allocates space for spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_alloc(spl,mx,nqty)

      integer, intent(in) :: mx,nqty
      type(spline_type), intent(inout) :: spl
      ! if(spl%allocated) call spline_dealloc(spl)
c-----------------------------------------------------------------------
c     safety check.
c-----------------------------------------------------------------------
      if(spl%allocated)
     $   call program_stop("spline_alloc: spline already allocated")

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
      spl%allocated=.true.
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_alloc
c-----------------------------------------------------------------------
c     subprogram 2. spline_dealloc.
c     deallocates space for spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_dealloc(spl)

      type(spline_type), intent(inout) :: spl
c-----------------------------------------------------------------------
c     safety check.
c-----------------------------------------------------------------------
      if(.not.spl%allocated)
     $   call program_stop("spline_dealloc: spline not allocated")
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
      if(allocated(spl%fsi))deallocate(spl%fsi)
      spl%allocated=.false.
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_dealloc
c-----------------------------------------------------------------------
c     subprogram 3. spline_fit.
c     switch between cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fit(spl,endmode)

      type(spline_type), intent(inout) :: spl
      integer, intent(in) :: endmode
c-----------------------------------------------------------------------
c     switch between two spline_fit.
c-----------------------------------------------------------------------
      if (use_classic_splines .and.
     $    (endmode  == 3 .OR. endmode == 1))then ! 3 = Extrapolate, 1 = Natural
         call spline_fit_classic(spl,endmode)
      else
         call spline_fit_ahg(spl,endmode)
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_fit
c-----------------------------------------------------------------------
c     subprogram 4. spline_fit_ahg.
c     fits real functions to cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fit_ahg(spl,endmode)

      type(spline_type), intent(inout) :: spl
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
      call spline_fac(spl,a,b,cl,cr,endmode)
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
         call spline_trilus(a(:,1:spl%mx-1),spl%fs1(1:spl%mx-1,:))
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      case(4) ! 4 = not-a-knot
         spl%fs1(1,:)=spl%fs1(1,:)-(2*spl%fs(1,:)
     $        -spl%fs(0,:)-spl%fs(2,:))*2*b(1)
         spl%fs1(spl%mx-1,:)=spl%fs1(spl%mx-1,:)
     $        +(2*spl%fs(spl%mx-1,:)-spl%fs(spl%mx,:)
     $        -spl%fs(spl%mx-2,:))*2*b(spl%mx)
         call spline_trilus(a(:,1:spl%mx-1),spl%fs1(1:spl%mx-1,:))
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
c     periodic boudary conditions.
c-----------------------------------------------------------------------
      case(2) ! 2 = Periodic
         spl%periodic=.true.
         spl%fs1(0,:)=3*((spl%fs(1,:)-spl%fs(0,:))*b(1)
     $        +(spl%fs(0,:)-spl%fs(spl%mx-1,:))*b(spl%mx))
         call spline_morrison(a(:,0:spl%mx-1),spl%fs1(0:spl%mx-1,:))
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
      end subroutine spline_fit_ahg
c-----------------------------------------------------------------------
c     subprogram 5. spline_fit_classic.
c     classical way to solve spline coefficients
c     does not support periodic and not-a-knot boundary conditions
c     g=a+b(x-xi)+c(x-xi)^2+d(x-xi)^3
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fit_classic(spl,endmode)
      type(spline_type), intent(inout) :: spl
      integer, intent(in) :: endmode
      real(r8), dimension(:),allocatable :: d,l,u,h
      real(r8), dimension(:,:),allocatable :: r
      real(r8), dimension(0:spl%mx) :: xfac

      integer :: iside,iqty,i
      real(r8),dimension(spl%nqty) :: bs,cs,ds
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
     $           -(spl%fs(i,:)-spl%fs(i-1,:))/h(i-1) )*6
      enddo
      r(spl%mx,:)=0

      if (endmode==3) then ! 3 = Extrapolate
         call spline_get_yp(spl%xs(0:3),spl%fs(0:3,:),
     $                      spl%xs(0),r(0,:),spl%nqty)
         call spline_get_yp(spl%xs(spl%mx-3:spl%mx),
     $        spl%fs(spl%mx-3:spl%mx,:),spl%xs(spl%mx),
     $        r(spl%mx,:),spl%nqty)
         d(0)=2*h(0)
         d(spl%mx)=2*h(spl%mx-1)
         u(1)=h(0)
         l(spl%mx)=h(spl%mx-1)
         r(0,:)=( (spl%fs(1,:)-spl%fs(0,:))/h(0) - r(0,:) )*6
         r(spl%mx,:)=( r(spl%mx,:)
     $  -(spl%fs(spl%mx,:)-spl%fs(spl%mx-1,:))/h(spl%mx-1) )*6

      endif
c-----------------------------------------------------------------------
c     solve and contrruct spline.
c-----------------------------------------------------------------------

      call spline_thomas(l,d,u,r,spl%mx+1,spl%nqty)

      do i=0, spl%mx-1
         bs=(spl%fs(i+1,:)-spl%fs(i,:))/h(i)
     $    - 0.5*h(i)*r(i,:)
     $    - h(i)*(r(i+1,:)-r(i,:))/6
         spl%fs1(i,:)=bs
      enddo
      ds=(r(spl%mx,:)-r(spl%mx-1,:))/(h(spl%mx-1)*6)
      cs=r(spl%mx-1,:)*0.5
      i=spl%mx-1
      bs=(spl%fs(i+1,:)-spl%fs(i,:))/h(i)
     $    - 0.5*h(i)*r(i,:)
     $    - h(i)*(r(i+1,:)-r(i,:))/6
      i=spl%mx
      spl%fs1(i,:)=bs+h(i-1)*(cs*2+h(i-1)*ds*3)
      deallocate (d,l,u,r,h)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_fit_classic
c-----------------------------------------------------------------------
c     subprogram 6. spline_fit_ha.
c     fits real functions to highly accurate cubic splines
c     using generalized matrix solves: sacrificing speed for accuracy
c     and generalizability
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fit_ha(spl,endmode)
      type(spline_type), intent(inout) :: spl
      integer, intent(in) :: endmode

      integer ::icount,icoef,imx,iqty,istart,jstart,info,iside
      integer :: ndim,nqty,kl,ku,ldab,nvar
      integer, dimension(:), allocatable :: ipiv
      integer,  dimension(:,:), allocatable :: imap
      real(r8) :: x0,x1,x2,dx
      real(r8), dimension(0:spl%mx) :: xfac
      real(r8), dimension(:,:), allocatable :: rhs,locrhs,fs,locrhs0,
     $                                         locrhs1,tmpmat
      real(r8), dimension(:,:),allocatable :: mat,locmat,locmat0,locmat1
      real(r8), dimension(:,:,:),allocatable :: coef
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
c     contruct grid and related matrix block.
c     five unknow variables at each interval a,b,c,e,f
c     a(x-xi)^3+b(x-xi)^2+c(x-xi)+d, where d=yi
c     e and f are used for passing periodic conditions
c-----------------------------------------------------------------------
      nqty=spl%nqty
      nvar=5
      allocate (coef(nvar,0:spl%mx-1,nqty),imap(nvar,0:spl%mx-1))
      allocate (fs(0:spl%mx,nqty))
      icount=1
      do imx=0, spl%mx-1
         do icoef=1,nvar
            imap(icoef,imx)=icount
            icount=icount+1
         enddo
      enddo
      ndim=icount-1
      kl=nvar*3
      ku=kl
      ldab=2*kl+ku+1
      allocate(mat(ldab,ndim),rhs(ndim,nqty),ipiv(ndim))
      mat=0
      rhs=0
      fs=spl%fs(:,:)
c-----------------------------------------------------------------------
c     contruct local matrix in each interval.
c-----------------------------------------------------------------------
      allocate(locmat(nvar,nvar*3),locmat0(nvar,nvar*3),
     &         locmat1(nvar,nvar*2),tmpmat(nvar,nvar*2))
      allocate(locrhs(nvar,nqty),locrhs0(nvar,nqty),locrhs1(nvar,nqty))
      do imx=1,spl%mx-2
         x0=spl%xs(imx-1)
         x1=spl%xs(imx)
         x2=spl%xs(imx+1)
         locmat=0
         locrhs=0
c-----------------------------------------------------------------------
c     S'0(x1)=S'1(x1)
c-----------------------------------------------------------------------
         dx=x1-x0
         locmat(1,1)=3*dx*dx
         locmat(1,2)=2*dx
         locmat(1,3)=1
         locmat(1,nvar+3)=-1
c-----------------------------------------------------------------------
c     S''1(x2)=S''2(x2)
c-----------------------------------------------------------------------
         dx=x2-x1
         locmat(2,nvar+1)=6*dx
         locmat(2,nvar+2)=2
         locmat(2,2*nvar+2)=-2
c-----------------------------------------------------------------------
c     S1(x2)=S2(x2)
c-----------------------------------------------------------------------
         locmat(3,nvar+1)=dx*dx*dx
         locmat(3,nvar+2)=dx*dx
         locmat(3,nvar+3)=dx
         locrhs(3,:)=fs(imx+1,:)-fs(imx,:)
c-----------------------------------------------------------------------
c     e1=e2
c-----------------------------------------------------------------------
         locmat(4,nvar+4)=1
         locmat(4,2*nvar+4)=-1
c-----------------------------------------------------------------------
c     f0=f1
c-----------------------------------------------------------------------
         locmat(5,5)=1
         locmat(5,nvar+5)=-1
c-----------------------------------------------------------------------
c     fill global matrix
c-----------------------------------------------------------------------
         istart=imap(1,imx)
         jstart=imap(1,imx-1)
         call spline_fill_matrix(mat,locmat,istart,jstart,kl,ku)
         rhs(istart:istart+nvar-1,:)=locrhs
      enddo
c-----------------------------------------------------------------------
c     boundary condition
c-----------------------------------------------------------------------
      locmat0=0
      locrhs0=0
      dx=spl%xs(1)-spl%xs(0)

      locmat0(2,nvar+1)=6*dx
      locmat0(2,nvar+2)=2
      locmat0(2,2*nvar+2)=-2

      locmat0(3,nvar+1)=dx*dx*dx
      locmat0(3,nvar+2)=dx*dx
      locmat0(3,nvar+3)=dx
      locrhs0(3,:)=fs(1,:)-fs(0,:)

      locmat0(4,nvar+4)=1
      locmat0(4,2*nvar+4)=-1

      locmat1=0
      locrhs1=0
      dx=spl%xs(spl%mx-1)-spl%xs(spl%mx-2)

      locmat1(1,1)=3*dx*dx
      locmat1(1,2)=2*dx
      locmat1(1,3)=1
      locmat1(1,nvar+3)=-1

      dx=spl%xs(spl%mx)-spl%xs(spl%mx-1)
      locmat1(2,nvar+1)=dx*dx*dx
      locmat1(2,nvar+2)=dx*dx
      locmat1(2,nvar+3)=dx
      locrhs1(2,:)=fs(spl%mx,:)-fs(spl%mx-1,:)

      locmat1(5,5)=1
      locmat1(5,nvar+5)=-1

      select case(endmode)
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      case(4) ! 4 = not-a-knot
         locmat0(1,nvar+1)=6
         locmat0(1,2*nvar+1)=-6
         locmat0(5,nvar+5)=1
         locrhs0(5,:)=1

         locmat1(3,1)=6
         locmat1(3,nvar+1)=-6

         locmat1(4,nvar+4)=1
         locrhs1(4,:)=1
c-----------------------------------------------------------------------
c     extrap boudary conditions, use first and last four points to
c     calculate y'(0) and y'(1).
c-----------------------------------------------------------------------
      case(3) ! 3 = Extrapolate
         locmat0(1,nvar+3)=1
         call spline_get_yp(spl%xs(0:3),spl%fs(0:3,:),
     $                      spl%xs(0),locrhs0(1,:),nqty)

         locmat0(5,nvar+5)=1
         locrhs0(5,:)=1

         dx=spl%xs(spl%mx)-spl%xs(spl%mx-1)
         locmat1(3,nvar+1)=3*dx*dx
         locmat1(3,nvar+2)=2*dx
         locmat1(3,nvar+3)=1
         call spline_get_yp(spl%xs(spl%mx-3:spl%mx),
     $   spl%fs(spl%mx-3:spl%mx,:),spl%xs(spl%mx),locrhs1(3,:),nqty)

         locmat1(4,nvar+4)=1
         locrhs1(4,:)=1
c-----------------------------------------------------------------------
c     natural boudary conditions.
c-----------------------------------------------------------------------
      case(1) ! 1 = Natural
         locmat0(1,nvar+2)=2

         locmat0(5,nvar+5)=1
         locrhs0(5,:)=1

         dx=spl%xs(spl%mx)-spl%xs(spl%mx-1)
         locmat1(3,nvar+1)=6*dx
         locmat1(3,nvar+2)=2

         locmat1(4,nvar+4)=1
         locrhs1(4,:)=1
c-----------------------------------------------------------------------
c     periodic boudary conditions.
c-----------------------------------------------------------------------
      case(2) ! 2 = Periodic
c-----------------------------------------------------------------------
c     s'0(x0)=s'm-1(xm).
c-----------------------------------------------------------------------
         do iqty=1,nqty
            if (ABS(spl%fs(0,iqty)-spl%fs(spl%mx,iqty)) > 1E-15) then
               write(*,*)
     $             "Warning: first and last points are different.
     $              IQTY= ",IQTY,",  averaged value is used."
c                    ,endmode
              spl%fs(0,iqty)=(spl%fs(0,iqty)+spl%fs(spl%mx,iqty))*0.5
              spl%fs(spl%mx,iqty)=spl%fs(0,iqty)
            endif
         enddo
         locmat0(1,nvar+3)=1
         locmat0(1,nvar+4)=-1

         dx=spl%xs(spl%mx)-spl%xs(spl%mx-1)
         locmat1(4,nvar+1)=3*dx*dx
         locmat1(4,nvar+2)=2*dx
         locmat1(4,nvar+3)=1
         locmat1(4,nvar+4)=-1
c-----------------------------------------------------------------------
c     s''0(x0)=s''m-1(xm).
c-----------------------------------------------------------------------
         locmat1(3,nvar+1)=6*dx
         locmat1(3,nvar+1)=2
         locmat1(3,nvar+5)=-1

         locmat0(5,nvar+2)=2
         locmat0(5,nvar+5)=-1
         spl%periodic=.true.
c-----------------------------------------------------------------------
c     unrecognized boundary condition.
c-----------------------------------------------------------------------
      case default
         call program_stop
     $       ("Cannot recognize endmode")
      end select
c-----------------------------------------------------------------------
c     fill global matrix at x0
c-----------------------------------------------------------------------
      istart=imap(1,0)
      jstart=imap(1,0)
      tmpmat=locmat0(:,nvar+1:3*nvar)
      call spline_fill_matrix(mat,tmpmat,istart,jstart,kl,ku)
      rhs(istart:istart+nvar-1,:)=locrhs0
c-----------------------------------------------------------------------
c     fill global matrix at xm
c-----------------------------------------------------------------------
      istart=imap(1,spl%mx-1)
      jstart=imap(1,spl%mx-2)
      tmpmat=locmat1(:,1:2*nvar)
      call spline_fill_matrix(mat,tmpmat,istart,jstart,kl,ku)
      rhs(istart:istart+nvar-1,:)=locrhs1
c-----------------------------------------------------------------------
c     solve global matrix
c-----------------------------------------------------------------------
      call dgbtrf(ndim,ndim,kl,ku,mat,ldab,ipiv,info)
      if (info .NE. 0) then
         call program_stop
     $           ("Error: LU factorization of spline matrix info ne 0")
      endif
      call dgbtrs("N",ndim,kl,ku,nqty,mat,ldab,ipiv,rhs,ndim,info )
      if (info .NE. 0) then
         call program_stop
     $        ("Error: solve spline matrix info ne 0")
      endif
c-----------------------------------------------------------------------
c     get spl%fs1.
c-----------------------------------------------------------------------
      do imx=0, spl%mx-1
         do icoef=1,nvar
            coef(icoef,imx,:)=rhs(imap(icoef,imx),:)
         enddo
         spl%fs1(imx,:)=coef(3,imx,:)
      enddo
      dx=spl%xs(imx)-spl%xs(imx-1)
      spl%fs1(imx,:)=coef(3,imx-1,:)
     $                   +dx*(coef(2,imx-1,:)*2+dx*coef(1,imx-1,:)*3)

      deallocate (coef,imap)
      deallocate (fs)
      deallocate(mat,rhs,ipiv)
      deallocate(locmat,locmat0,locmat1,tmpmat)
      deallocate(locrhs,locrhs0,locrhs1)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_fit_ha
c-----------------------------------------------------------------------
c     subprogram 7. spline_fac.
c     sets up matrix for cubic spline fitting.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fac(spl,a,b,cl,cr,endmode)

      type(spline_type), intent(in) :: spl
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
         call spline_triluf(a(:,1:spl%mx-1))
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
         call spline_triluf(a(:,1:spl%mx-1))
c-----------------------------------------------------------------------
c     periodic boundary conditions.
c-----------------------------------------------------------------------
      case(2) ! 2 = Periodic
         a(0,0:spl%mx:spl%mx)=2*(b(spl%mx)+b(1))
         a(1,0)=b(1)
         a(-1,0)=b(spl%mx)
         b=b*b
         call spline_sherman(a(:,0:spl%mx-1))
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
      end subroutine spline_fac
c-----------------------------------------------------------------------
c     subprogram 8. spline_eval.
c     evaluates real cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_eval(spl,x,mode)

      type(spline_type), intent(inout) :: spl
      real(r8), intent(in) :: x
      integer, intent(in) :: mode

      integer :: iqty,iside
      real(r8) :: xx,d,z,z1,xfac,dx
      real(r8) :: g,g1,g2,g3
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
         if(xx >= spl%xs(spl%ix).OR.spl%ix <= 0)EXIT
         spl%ix=spl%ix-1
      enddo
      do
         if(xx < spl%xs(spl%ix+1).OR.spl%ix >= spl%mx-1)EXIT
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
      spl%f=spl%fs(spl%ix,1:spl%nqty)*z1*z1*(3-2*z1)
     $     +spl%fs(spl%ix+1,1:spl%nqty)*z*z*(3-2*z)
     $     +d*z*z1*(spl%fs1(spl%ix,1:spl%nqty)*z1
     $     -spl%fs1(spl%ix+1,1:spl%nqty)*z)
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      if(mode > 0)then
         spl%f1=6*(spl%fs(spl%ix+1,1:spl%nqty)
     $        -spl%fs(spl%ix,1:spl%nqty))*z*z1/d
     $        +spl%fs1(spl%ix,1:spl%nqty)*z1*(3*z1-2)
     $        +spl%fs1(spl%ix+1,1:spl%nqty)*z*(3*z-2)
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      if(mode > 1)then
         spl%f2=(6*(spl%fs(spl%ix+1,1:spl%nqty)
     $        -spl%fs(spl%ix,1:spl%nqty))*(z1-z)/d
     $        -spl%fs1(spl%ix,1:spl%nqty)*(6*z1-2)
     $        +spl%fs1(spl%ix+1,1:spl%nqty)*(6*z-2))/d
      endif
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      if(mode > 2)then
         spl%f3=(12*(spl%fs(spl%ix,1:spl%nqty)
     $        -spl%fs(spl%ix+1,1:spl%nqty))/d
     $        +6*(spl%fs1(spl%ix,1:spl%nqty)
     $        +spl%fs1(spl%ix+1,1:spl%nqty)))/(d*d)
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
      end subroutine spline_eval
c-----------------------------------------------------------------------
c     subprogram 9. spline_eval_external
c     evaluates real cubic spline with external arrays (parallel).
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_eval_external(spl,x,s_f,s_f1,s_f2,s_f3)

      type(spline_type), intent(in) :: spl
      real(r8), intent(in) :: x

      integer :: iqty,iside
      integer :: ix, i_low, i_high, i_mid
      real(r8) :: xx,d,z,z1,xfac,dx
      real(r8) :: g,g1,g2,g3


      real(r8), dimension(:), intent(out) :: s_f
      real(r8), dimension(:), optional, intent(out) :: s_f1,s_f2,s_f3

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
      s_f=spl%fs(ix,1:spl%nqty)*z1*z1*(3-2*z1)
     $     +spl%fs(ix+1,1:spl%nqty)*z*z*(3-2*z)
     $     +d*z*z1*(spl%fs1(ix,1:spl%nqty)*z1
     $     -spl%fs1(ix+1,1:spl%nqty)*z)
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      if(present(s_f1))then
         s_f1=6*(spl%fs(ix+1,1:spl%nqty)
     $        -spl%fs(ix,1:spl%nqty))*z*z1/d
     $        +spl%fs1(ix,1:spl%nqty)*z1*(3*z1-2)
     $        +spl%fs1(ix+1,1:spl%nqty)*z*(3*z-2)
      endif
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      if(present(s_f2))then
         s_f2=(6*(spl%fs(ix+1,1:spl%nqty)
     $        -spl%fs(ix,1:spl%nqty))*(z1-z)/d
     $        -spl%fs1(ix,1:spl%nqty)*(6*z1-2)
     $        +spl%fs1(ix+1,1:spl%nqty)*(6*z-2))/d
      endif
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      if(present(s_f3))then
         s_f3=(12*(spl%fs(ix,1:spl%nqty)
     $        -spl%fs(ix+1,1:spl%nqty))/d
     $        +6*(spl%fs1(ix,1:spl%nqty)
     $        +spl%fs1(ix+1,1:spl%nqty)))/(d*d)
      endif
c-----------------------------------------------------------------------
c     restore powers.
c-----------------------------------------------------------------------
      do iside=1,2
         dx=x-spl%x0(iside)
         do iqty=1,spl%nqty
            if(spl%xpower(iside,iqty) == 0)cycle
            xfac=ABS(dx)**spl%xpower(iside,iqty)
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
      end subroutine spline_eval_external
c-----------------------------------------------------------------------
c     subprogram 10. spline_all_eval.
c     evaluates cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_all_eval(spl,z,f,f1,f2,f3,mode)

      type(spline_type), intent(inout) :: spl
      real(r8), intent(in) :: z
      real(r8), dimension(spl%mx,spl%nqty), intent(out) :: f,f1,f2,f3
      integer, intent(in) :: mode

      integer :: iqty,nqty,n,iside
      real(r8) :: z1
      real(r8), dimension(spl%mx) :: d,xfac,dx
      real(r8), dimension(spl%mx) :: g,g1,g2,g3
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
      end subroutine spline_all_eval
c-----------------------------------------------------------------------
c     subprogram 11. spline_write1.
c     produces ascii and binary output for real cubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_write1(spl,out,bin,iua,iub,interp)

      type(spline_type), intent(inout) :: spl
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      logical, intent(in) :: interp

      character(30) :: format1,format2
      integer :: i,j
      real(r8) :: x,dx
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   format('(/3x,"ix",',i2.2,'(4x,a6,1x)/)')
 20   format('(i5,1p,',i2.2,'e11.3)')
!  30   format('(/3x,"ix",2x,"j",',i2.2,'(4x,a6,1x)/)')
c-----------------------------------------------------------------------
c     abort.
c-----------------------------------------------------------------------
      if(.not.out.and..not.bin)return
c-----------------------------------------------------------------------
c     print node values.
c-----------------------------------------------------------------------
      if(out)then
         write(format1,10)spl%nqty+1
         write(format2,20)spl%nqty+1
         write(iua,'(/1x,a)')'node values:'
         write(iua,format1)spl%title(0:spl%nqty)
      endif
      do i=0,spl%mx
         call spline_eval(spl,spl%xs(i),0)
         if(out)write(iua,format2)i,spl%xs(i),spl%f
         if(bin)write(iub)real(spl%xs(i),4),real(spl%f,4)
      enddo
      if(out)write(iua,format1)spl%title(0:spl%nqty)
      if(bin)write(iub)
      if(.not. interp)return
c-----------------------------------------------------------------------
c     print header for interpolated values.
c-----------------------------------------------------------------------
      if(out)then
         write(iua,'(/1x,a)')'interpolated values:'
         write(iua,format1)spl%title(0:spl%nqty)
      endif
c-----------------------------------------------------------------------
c     print interpolated values.
c-----------------------------------------------------------------------
      do i=0,spl%mx-1
         dx=(spl%xs(i+1)-spl%xs(i))/4
         do j=0,4
            x=spl%xs(i)+j*dx
            call spline_eval(spl,x,0)
            if(out)write(iua,format2)i,x,spl%f
            if(bin)write(iub)real(x,4),real(spl%f,4)
         enddo
      enddo
c-----------------------------------------------------------------------
c     print final interpolated values.
c-----------------------------------------------------------------------
      x=spl%xs(spl%mx)
      call spline_eval(spl,x,0)
      if(out)then
         write(iua,format2)i,x,spl%f
         write(iua,format1)spl%title
      endif
      if(bin)then
         write(iub)real(x,4),real(spl%f,4)
         write(iub)
         call bin_close(bin_unit)
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_write1
c-----------------------------------------------------------------------
c     subprogram 12. spline_write2.
c     produces ascii and binary output for real cubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_write2(spl,out,bin,iua,iub,interp)

      type(spline_type), intent(inout) :: spl
      logical, intent(in) :: out,bin
      integer, intent(in) :: iua,iub
      logical, intent(in) :: interp

      character(30) :: format1,format2
      integer :: i,j,iz
      real(r8) :: x,dx,z
      real(r8), dimension(spl%mx,spl%nqty) :: f, f1,f2,f3
      real(r8), dimension(0:4*spl%mx,spl%nqty) :: g
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   format('(/4x,"i",',i2.2,'(4x,a6,1x)/)')
 20   format('(i5,1p,',i2.2,'e11.3)')
!  30   format('(/4x,"i",2x,"j",',i2.2,'(4x,a6,1x)/)')
c-----------------------------------------------------------------------
c     compute values.
c-----------------------------------------------------------------------
      z=0
      do iz=0,3
         call spline_all_eval(spl,z,f,f1,f2,f3,0)
         g(iz:4*spl%mx-1:4,:)=f
         z=z+.25_r8
      enddo
      call spline_eval(spl,spl%xs(spl%mx),0)
      g(4*spl%mx,:)=spl%f
c-----------------------------------------------------------------------
c     print node values.
c-----------------------------------------------------------------------
      if(.not.out.and..not.bin)return
      if(out)then
         write(format1,10)spl%nqty+1
         write(format2,20)spl%nqty+1
         write(iua,'(/1x,a)')'node values:'
         write(iua,format1)spl%title(0:spl%nqty)
      endif
      do i=0,spl%mx
         if(out)write(iua,format2)i,spl%xs(i),g(4*i,:)
         if(bin)write(iub)real(spl%xs(i),4),real(g(4*i,:),4)
      enddo
      if(out)write(iua,format1)spl%title(0:spl%nqty)
      if(bin)write(iub)
c-----------------------------------------------------------------------
c     print header for interpolated values.
c-----------------------------------------------------------------------
      if(.not. interp)return
      if(out)then
         write(iua,'(/1x,a)')'interpolated values:'
         write(iua,format1)spl%title(0:spl%nqty)
      endif
c-----------------------------------------------------------------------
c     print interpolated values.
c-----------------------------------------------------------------------
      do i=0,spl%mx-1
         dx=(spl%xs(i+1)-spl%xs(i))/4
         do j=0,4
            x=spl%xs(i)+j*dx
            if(out)write(iua,format2)i,x,g(4*i+j,:)
            if(bin)write(iub)real(x,4),real(g(4*i+j,:),4)
         enddo
         if(out)write(iua,'(1x)')
      enddo
c-----------------------------------------------------------------------
c     print final interpolated values.
c-----------------------------------------------------------------------
      if(out)then
         write(iua,format2)i,x,g(spl%mx*4,:)
         write(iua,format1)spl%title(0:spl%nqty)
      endif
      if(bin)then
         write(iub)real(x,4),real(g(spl%mx*4,:),4)
         write(iub)
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_write2
c-----------------------------------------------------------------------
c     subprogram 13. spline_int.
c     integrates real cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_int(spl)

      type(spline_type), intent(inout) :: spl

      integer :: ix,iqty,ig
      real(r8), dimension(spl%mx) :: dx
      real(r8), dimension(spl%mx,spl%nqty) :: term,f,f1,f2,f3

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
      if(.not.allocated(spl%fsi))allocate(spl%fsi(0:spl%mx,spl%nqty))
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
               call spline_all_eval(spl,xg(ig),f,f1,f2,f3,0)
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
      end subroutine spline_int
c-----------------------------------------------------------------------
c     subprogram 14. spline_triluf.
c     performs tridiagonal LU factorization.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_triluf(a)

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
      end subroutine spline_triluf
c-----------------------------------------------------------------------
c     subprogram 15. spline_trilus.
c     performs tridiagonal LU solution.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_trilus(a,x)

      real(r8), dimension(-1:,:), intent(in) :: a
      real(r8), dimension(:,:), intent(inout) :: x

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
      end subroutine spline_trilus
c-----------------------------------------------------------------------
c     subprogram 16. spline_sherman.
c     uses Sherman-Morrison formula to factor periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_sherman(a)

      real(r8), dimension(-1:,:), intent(inout) :: a

      integer :: j,n
      real(r8), dimension(SIZE(a,2),1) :: u
c-----------------------------------------------------------------------
c     prepare matrices.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      a(0,1)=a(0,1)-a(-1,1)
      a(0,n)=a(0,n)-a(-1,1)
      u=RESHAPE((/one,(zero,j=2,n-1),one/),SHAPE(u))
      call spline_triluf(a)
      call spline_trilus(a,u)
      a(-1,1)=a(-1,1)/(1+a(-1,1)*(u(1,1)+u(n,1)))
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_sherman
c-----------------------------------------------------------------------
c     subprogram 17. spline_morrison.
c     uses Sherman-Morrison formula to solve periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_morrison(a,x)

      real(r8), dimension(-1:,:), intent(in) :: a
      real(r8), dimension(:,:), intent(inout) :: x

      integer :: n
      real(r8), dimension(SIZE(x,1),SIZE(x,2)) :: y
c-----------------------------------------------------------------------
c     solve for x.
c-----------------------------------------------------------------------
      n=SIZE(a,2)
      y=x
      call spline_trilus(a,y)
      x(1,:)=x(1,:)-a(-1,1)*(y(1,:)+y(n,:))
      x(n,:)=x(n,:)-a(-1,1)*(y(1,:)+y(n,:))
      call spline_trilus(a,x)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_morrison
c-----------------------------------------------------------------------
c     subprogram 18. spline_copy.
c     copies one spline_type to another.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_copy(spl1,spl2)

      type(spline_type), intent(in) :: spl1
      type(spline_type), intent(inout) :: spl2
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      if(allocated(spl2%xs))call spline_dealloc(spl2)
      call spline_alloc(spl2,spl1%mx,spl1%nqty)
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
      end subroutine spline_copy
c-----------------------------------------------------------------------
c     subprogram 19. spline_fill_matrix.
c     fill local matrix into global matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_fill_matrix(mat,locmat,istart,jstart,kl,ku)
      integer :: istart,jstart,itot,jtot,i,j,kl,ku,m,n,offset
      real(r8), dimension(:,:), allocatable,intent(inout) :: mat
      real(r8), dimension(:,:), allocatable,intent(in) :: locmat
c-----------------------------------------------------------------------
c     fill matrix.
c-----------------------------------------------------------------------
      itot=SIZE(locmat,1)
      jtot=SIZE(locmat,2)
      offset=kl+ku+1
      do m=1, itot
         i=m+istart-1
         do n=1, jtot
            j=n+jstart-1
            mat(offset+i-j,j)=mat(offset+i-j,j)+locmat(m,n)
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_fill_matrix
c-----------------------------------------------------------------------
c     subprogram 20. spline_get_yp.
c     get yi' with four points for spline boundary condtion .
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_get_yp(x,y,xi,yip,nqty)
      integer, intent(in) :: nqty
      integer :: n,nrhs,lda,info,ldb,i
      integer, dimension(4) :: ipiv
      real(r8) :: dx
      real(r8), intent(in) :: xi
      real(r8), dimension(nqty),intent(out) :: yip
      real(r8), dimension(4), intent(in) :: x
      real(r8), dimension(4,nqty), intent(in) :: y
      real(r8), dimension(4,nqty) :: b
      real(r8), dimension(4,4) :: a
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
      call dgesv(n,nrhs,a,lda,ipiv,b,ldb,info)
      dx=xi-x(1)
      yip=(3*b(1,:)*dx+2*b(2,:))*dx+b(3,:)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_get_yp
c-----------------------------------------------------------------------
c     subprogram 21. spline_thomas.
c     thomas method to solve tri-diagnol matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_thomas(l,d,u,b,n,m)

      integer, intent(in):: n,m
      real(r8), dimension(n), intent(inout):: d
      real(r8), dimension(n-1), intent(inout):: l,u
      real(r8), dimension(n,m), intent(inout):: b
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
      end subroutine spline_thomas
c-----------------------------------------------------------------------
c     subprogram 22. spline_roots.
c     Calculate all the roots of a cubic spline.
c     Necessary because simple iterative techniques may miss cases with
c     multiple roots between knots.
c     Analytics from https://www.e-education.psu.edu/png520/m11_p6.html
c
c     Note: Doesn't handle powers (what are those?)
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_roots(spl, iqty, nroots, roots,op_extrap,op_eps)

      type(spline_type), intent(inout) :: spl
      integer, intent(in):: iqty
      integer, intent(out) :: nroots
      real(r8), dimension(*), intent(out):: roots
      !^^^^ should be dimension(1:spl%mx*3) ^^^^
      logical, intent(in), optional :: op_extrap
      real(r8), intent(in), optional :: op_eps

      logical, PARAMETER :: debug = .false.

      logical :: extrap
      integer:: ix, jroot, lx, nzvalid
      real(r8)::  f_0, f1_0, f_1, f1_1,
     $  c3, c2, c1, c0, a, b, c, q, r, m, s, t, theta,
     $  z1, z2, z3, x1, x2, x3, last1, last2, last3,
     $  dx, eps, tol
      integer, dimension(:), allocatable :: index
      real(r8), dimension(:), allocatable :: tmproots

      ! allow optional accuracy manipulation repeated solutions
      ! within eps*stepsize are rejected
      if(present(op_eps))then
         eps = op_eps
      else
         eps = 1e-6
      endif

      ! allow optional extrapolation beyond ends of the spline
      if(present(op_extrap))then
         extrap = op_extrap
      else
         extrap = .false.
      endif

      roots(1:spl%mx*3) = spl%xs(0) - HUGE(last1)
      last1 = spl%xs(0) - HUGE(last1)
      last2 = spl%xs(0) - HUGE(last2)
      last3 = spl%xs(0) - HUGE(last3)
      lx = spl%mx-1
      jroot = 0
      ! find the analytic roots of a cubic polynomial between each knot
      do ix=0, lx
         nzvalid = 0
         ! step size and associated tolerance for repeated roots
         dx=spl%xs(ix+1)-spl%xs(ix)
         tol = eps * dx
         ! get the value and derivatives for this step span
         f_0 = spl%fs(ix,iqty)
         f1_0 = spl%fs1(ix,iqty)
         f_1 = spl%fs(ix+1,iqty)
         f1_1 = spl%fs1(ix+1,iqty)
         ! starting from the equation for f(x) in spline_eval,
         ! we reorganize into a classic c3 z^3 + c2 z^2 + c1 z + c0
         c3 = -2*f_0 - 2*f_1 +   dx*f1_0 + dx*f1_1
         c2 =  5*f_0 + 3*f_1 - 2*dx*f1_0 - dx*f1_1
         c1 = -4*f_0 + dx*f1_0
         c0 = f_0
         c3 =  2*f_0 - 2*f_1 +   dx*f1_0 - dx*f1_1
         c2 = -3*f_0 + 3*f_1 - 2*dx*f1_0 + dx*f1_1
         c1 = dx*f1_0
         c0 = f_0
         c3 =  2*f_0 - 2*f_1 +   dx*f1_0 + dx*f1_1
         c2 = -3*f_0 + 3*f_1 - 2*dx*f1_0 - dx*f1_1
         c1 = dx*f1_0
         c0 = f_0
         ! we have to watch out for secret reductions!
         if(c3==0)then
            if(debug) PRinT *, "  >> Spline quadratic @",spl%xs(ix)
            a = c2
            b = c1
            c = c0
            if(b**2 - 4.0*a*c >= 0)then
               z1 = (-b + sqrt(b**2 - 4.0*a*c)) / (2.0 * a)
               z2 = (-b - sqrt(b**2 - 4.0*a*c)) / (2.0 * a)
               nzvalid = 2
            elseif(a==0)then
               if(debug) PRinT *, "  >> Spline linear @",spl%xs(ix)
               z1 = -c / b
               z2 = -huge(z2)
               nzvalid = 1
            elseif(a==0 .and. b==0 .and. c==0)then
               if(debug) PRinT *, "  >> Spline all 0 @",spl%xs(ix)
               z1 = 0
               nzvalid = 1
            else
               z1 = -huge(z1)
               z2 = -huge(z2)
               nzvalid = 0
            endif
           z3 = -huge(z3)
         else
            ! truely a cubic case
            ! since we want f = 0 we can normalize out the c3
            ! so 0 = z^3 + a z^2 + b z + c
            a = c2 / c3
            b = c1 / c3
            c = c0 / c3
            ! now we just get the analytic solutions
            q = (a * a - 3.0 * b) / 9.0
            r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
            m = r**2 - q**3  ! discrimenent
            if(m > 0)then ! one real root
               s = -sign(1.0_r8, r) * (abs(r) + sqrt(m))**(1.0/3.0)
               t = q / s
               z1 = s + t - (a / 3.0)
               z2 = -huge(z2)
               z3 = -huge(z3)
               nzvalid = 1
            else  ! three real roots (watch out for repeates)
               theta = acos(r / sqrt(q**3))
               z1 = -2*sqrt(q)*cos(theta/3.0) - a/3.0
               z2 = -2*sqrt(q)*cos((theta+twopi)/3.0) - a/3.0
               z3 = -2*sqrt(q)*cos((theta-twopi)/3.0) - a/3.0
               nzvalid = 3
               if(abs(z2-z3)<tol) nzvalid = 2  ! double root
            endif
         endif
         ! note spline_eval uses step-size normalized coordinate z,
         ! but we want real x values
         x1 = spl%xs(ix) + z1 * dx
         x2 = spl%xs(ix) + z2 * dx
         x3 = spl%xs(ix) + z3 * dx
         ! check if they are in the knot interval
         ! (exclude repeats if periodic)
         if(debug .and. (f_0*f_1 <= 0))then
            PRinT *," > f zero crossing between",spl%xs(ix),spl%xs(ix+1)
            PRinT *,"  >> f_0, f_1 =",f_0,f_1
            PRinT *,"  >> nzvalid =",nzvalid
            PRinT *,"  >> z1, z2, z3 = ",z1,z2,z3
            PRinT *,"  >> c3,c2,c1,c0 =",c3,c2,c1,c0
         endif
         if(nzvalid>0 .and. (z1>-eps .OR. (ix==0 .and. extrap)))then
            if(z1<=1+eps .OR. (ix==lx .and. extrap)) then
               if(debug) PRinT '(a12,es17.8e3,a4,es17.8e3,a15,'//
     $            'I3,a1,I3,a11,es13.4e3,a1,es13.4)',
     $            "  > Found z1",z1,", x1",x1," between knots ",ix,",",
     $            ix+1," where x =",spl%xs(ix),",",spl%xs(ix+1)
               ! refine solution numerically
               ! (analytics vs reality of interp)
               call spline_refine_root(spl,iqty,x1)
               ! avoid repeated left/right solutions at a f=0 knot
               if(abs(last1-x1)>tol .and. abs(last2-x1)>tol .and.
     $            abs(last3-x1)>tol) then
                  jroot = jroot + 1
                  roots(jroot) = x1
               endif
            endif
         endif
         if(nzvalid>1 .and. (z2>-eps .OR. (ix==0 .and. extrap)))then
            if(z2<=1+eps .OR. (ix==lx .and. extrap)) then
               if(debug) PRinT '(a12,es17.8e3,a4,es17.8e3,a15,'//
     $            'I3,a1,I3,a11,es13.4e3,a1,es13.4)',
     $            "  > Found z2",z2,", x2",x2," between knots ",ix,
     $            ",",ix+1," where x =",spl%xs(ix),",",spl%xs(ix+1)
               ! refine solution numerically
               ! (analytics vs reality of interp)
               call spline_refine_root(spl,iqty,x2)
               ! avoid double roots or repeats right at a knot location
               if(abs(last1-x2)>tol .and. abs(last2-x2)>tol .and.
     $            abs(last3-x2)>tol .and. abs(x1-x2)>tol) then
                  jroot = jroot + 1
                  roots(jroot) = x2
               endif
            endif
         endif
         if(nzvalid>2 .and. (z3>-eps .OR. (ix==0 .and. extrap)))then
            if(z3<=1+eps .OR. (ix==lx .and. extrap)) then
               if(debug) PRinT '(a12,es17.8e3,a4,es17.8e3,a15,'//
     $            'I3,a1,I3,a11,es13.4e3,a1,es13.4)',
     $            "  > Found z3",z3,", x3",x3," between knots ",ix,
     $            ",",ix+1," where x =",spl%xs(ix),",",spl%xs(ix+1)
               ! refine solution numerically
               ! (analytics vs reality of interp)
               call spline_refine_root(spl,iqty,x3)
               ! avoid double roots or repeats right at a knot location
               if(abs(last1-x3)>tol .and. abs(last2-x3)>tol .and.
     $            abs(last3-x3)>tol .and. abs(x1-x3)>tol .and.
     $            abs(x2-x3)>tol) then
                  jroot = jroot + 1
                  roots(jroot) = x3
               endif
            endif
         endif
         last1 = roots(max(1,jroot))
         last2 = roots(max(1,jroot-1))
         last3 = roots(max(1,jroot-2))
      enddo
      nroots = jroot
      ! sort the roots lowest to highest
      allocate(index(nroots), tmproots(nroots))
      index=(/(ix,ix=1,nroots)/)
      tmproots(1:nroots) = roots(1:nroots)
      call bubble_sort(tmproots,index,1,nroots)
      do ix=1,nroots
         tmproots(ix) = roots(index(nroots + 1 - ix))
      enddo
      roots(1:nroots) = tmproots(1:nroots)
      if(nroots>0 .and. debug)
     $   PRinT *,"  > Sorted roots are",roots(1:nroots)
      deallocate(index, tmproots)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_roots

c-----------------------------------------------------------------------
c     subprogram 23. spline_refine_root.
c     Calculate the root of a spline using newton iteration from an
c     initial guess.
c     Doesn't handle powers (what are those?)
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine spline_refine_root(spl, iqty, x, op_tol)

      type(spline_type), intent(inout) :: spl
      integer, intent(in):: iqty
      real(r8), intent(inout) :: x
      real(r8), intent(in), optional :: op_tol
      ! declare variables
      integer :: it
      integer, PARAMETER :: itmax=500
      real(r8) :: tol=1e-12
      real(r8) :: dx,lx,lf,f,df

      ! set optional tolerance
      tol = 1e-12
      if(present(op_tol)) tol = op_tol

      ! if its exact, we don't need to do anything
      call spline_eval(spl,x,1)
      if(spl%f(iqty) == 0) return
      ! otherwise, we'll iterate
      lx=spl%xs(spl%ix+1)-spl%xs(spl%ix)  ! note ix set in above eval
      lf = maxval(spl%fs(:,iqty))-minval(spl%fs(:,iqty))
      dx=lx
      f=huge(f)
      it=0
      do
         call spline_eval(spl,x,1)
         df=spl%f(iqty)-f
        !if(abs(dx) < tol*lx .OR. abs(df) < tol*lf .OR. it >= itmax)EXIT
         if(abs(dx) <= abs(tol*lx) .OR. it >= itmax)EXIT
         it=it+1
         f=spl%f(iqty)
         dx=-spl%f(iqty)/spl%f1(iqty)
         x=x+dx
      enddo
      ! abort on failure.
      if(it >= itmax)then
         write(*,*)
         write(*,*) "!! WARNinG: root refining convergence failure"
         write(*,*) " -> estimated root ",x,
     $              " has error/tol ",abs(dx)/(tol*lx),abs(df)/(tol*lf)
         write(*,*)
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_refine_root
c-----------------------------------------------------------------------
c     subprogram 24. bubble_sort.
c     performs a bubble sort in decreasing order of value.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bubble_sort(key,index,mmin,mmax)

      real(r8), dimension(:), intent(in) :: key
      integer, dimension(:), intent(inout) :: index
      integer :: mmin,mmax

      logical :: switch
      integer :: i,temp
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      switch= .true.
      do while(switch)
         switch= .false.
         do i=mmin,mmax-1
            if(key(index(i)) < key(index(i+1)))then
               temp=index(i)
               index(i)=index(i+1)
               index(i+1)=temp
               switch= .true.
            endif
         enddo
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bubble_sort
c-----------------------------------------------------------------------
      end module spline_mod
