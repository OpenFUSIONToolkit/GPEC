c-----------------------------------------------------------------------
c     file fspline.f.
c     fits functions to cubic spline in x and Fourier series in y.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. fspline_mod.
c     1. fspline_alloc.
c     2. fspline_dealloc.
c     3. fspline_fit_1.
c     4. fspline_fit_2.
c     5. fspline_eval.
c     6. fspline_eval_external
c     7. fspline_all_eval.
c     8. fspline_write_xy.
c     9. fspline_write_yx.
c     10. fspline_copy.
c-----------------------------------------------------------------------
c     subprogram 0. fspline_type definition.
c     defines fspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      module fspline_mod
      use defs_mod
      use spline_mod
      use cspline_mod
      use fft_mod
      implicit none

      type :: fspline_type
      integer :: mx,my,mband,nqty
      real(r8), dimension(:), POinTER :: xs,ys
      real(r8), dimension(:,:,:), POinTER :: fs
      type(cspline_type) :: cs
      real(r8), dimension(:), POinTER :: f,fx,fy,fxx,fxy,fyy
      character(6) :: xtitle,ytitle
      character(6), dimension(:), POinTER :: title
      character(6) :: name
      end type fspline_type

      contains
c-----------------------------------------------------------------------
c     subprogram 1. fspline_alloc.
c     allocates space for fspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_alloc(fst,mx,my,mband,nqty)

      integer, intent(in) :: mx,my,mband,nqty
      type(fspline_type), intent(out) :: fst
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      fst%mx=mx
      fst%my=my
      fst%mband=mband
      fst%nqty=nqty
      allocate(fst%xs(0:mx))
      allocate(fst%ys(0:my))
      allocate(fst%fs(0:mx,0:my,nqty))
      allocate(fst%title(nqty))
      allocate(fst%f(nqty))
      allocate(fst%fx(nqty))
      allocate(fst%fy(nqty))
      allocate(fst%fxx(nqty))
      allocate(fst%fxy(nqty))
      allocate(fst%fyy(nqty))
      call cspline_alloc(fst%cs,mx,(mband+1)*nqty)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_alloc
c-----------------------------------------------------------------------
c     subprogram 2. fspline_dealloc.
c     deallocates space for fspline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_dealloc(fst)

      type(fspline_type), intent(inout) :: fst
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      deallocate(fst%xs)
      deallocate(fst%ys)
      deallocate(fst%fs)
      deallocate(fst%title)
      deallocate(fst%f)
      deallocate(fst%fx)
      deallocate(fst%fy)
      deallocate(fst%fxx)
      deallocate(fst%fxy)
      deallocate(fst%fyy)
      call cspline_dealloc(fst%cs)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_dealloc
c-----------------------------------------------------------------------
c     subprogram 3. fspline_fit_1.
c     fits functions to fsplines by integrating periodic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_fit_1(fst,endmode,fit_flag)

      type(fspline_type), intent(inout) :: fst
      integer, intent(in) :: endmode
      logical, intent(in) :: fit_flag

      integer :: m,ix,iq,mx,my,mband,nqty,j
      real(r8), PARAMETER :: eps=1e-3
      real(r8), allocatable, dimension(:) :: delta,d4fac,delta4
      real(r8), allocatable, dimension(:,:,:) :: fs,fsy
      complex(r8), allocatable, dimension(:) :: expfac0,expfac
      complex(r8), allocatable, dimension(:) :: dexpfac0,dexpfac
      complex(r8), allocatable, dimension(:) :: alpha1,alpha2,
     $                                          beta1,beta2
      complex(r8), allocatable, dimension(:,:,:) :: coef
      type(spline_type) :: spl
c-----------------------------------------------------------------------
c     allocate.
c-----------------------------------------------------------------------
      allocate (delta(fst%my),d4fac(fst%my),delta4(fst%my))
      allocate (fs(0:fst%mx,0:fst%my,fst%nqty),
     $          fsy(0:fst%mx,0:fst%my,fst%nqty))
      allocate (expfac0(0:fst%my),expfac(0:fst%my))
      allocate (dexpfac0(fst%my),dexpfac(fst%my))
      allocate (alpha1(fst%my),alpha2(fst%my),
     $          beta1(fst%my),beta2(fst%my))
      allocate (coef(0:fst%mx,0:fst%mband,fst%nqty))
c-----------------------------------------------------------------------
c     copy sizes and zero Fourier coefficients.
c-----------------------------------------------------------------------
      mx=fst%mx
      my=fst%my
      mband=fst%mband
      nqty=fst%nqty
      coef=0
c-----------------------------------------------------------------------
c     prepare principal periodic functions.
c-----------------------------------------------------------------------
      delta=fst%ys(1:my)-fst%ys(0:my-1)
      delta4=delta**4
      d4fac=1/delta4
      do ix=0,my
         expfac0(ix)=EXP(-ifac*fst%ys(ix))
      enddo
      dexpfac0=expfac0(0:my-1)/expfac0(1:my)
      expfac=1
      dexpfac=1
c-----------------------------------------------------------------------
c     compute y derivatives.
c-----------------------------------------------------------------------
      call spline_alloc(spl,my,mx+1)
      spl%xs=fst%ys
      do iq=1,nqty
         spl%fs=TRANSPOSE(fst%fs(:,:,iq))
         call spline_fit(spl,2)
         fsy(:,:,iq)=TRANSPOSE(spl%fs1)
      enddo
      fs=fst%fs
      call spline_dealloc(spl)
c-----------------------------------------------------------------------
c     compute alpha's and beta's.
c-----------------------------------------------------------------------
      do m=0,mband
         WHERE(ABS(m*delta) > eps)
            alpha1=(dexpfac*(12._r8-ifac*m*delta*(6._r8+(m*delta)**2))
     $           -6._r8*(2._r8+ifac*m*delta))*d4fac/m**4
            beta1=(dexpfac*(6._r8-m*delta*(4*ifac+m*delta))
     $           -2._r8*(3._r8+ifac*m*delta))*d4fac/m**4
         elseWHERE
            alpha1=.5_r8+7._r8*ifac*m*delta/20._r8-2._r8*
     $           (m*delta)**2/15._r8
            beta1=1._r8/12._r8+ifac*m*delta/20._r8-(m*delta)**2/60._r8
         endWHERE
         alpha1=alpha1*delta/twopi
         beta1=beta1*delta**2/twopi
         alpha2=CONJG(alpha1)*expfac(0:my-1)
         beta2=CONJG(beta1)*expfac(0:my-1)
         alpha1=alpha1*expfac(1:my)
         beta1=beta1*expfac(1:my)
c-----------------------------------------------------------------------
c     compute Fourier coefficients.
c-----------------------------------------------------------------------
         do iq=1,nqty
            do ix=0,mx
               coef(ix,m,iq)=coef(ix,m,iq)+SUM(
     $              alpha1*fs(ix,0:my-1,iq)+alpha2*fs(ix,1:my,iq)
     $              +beta1*fsy(ix,0:my-1,iq)-beta2*fsy(ix,1:my,iq))
            enddo
         enddo

c-----------------------------------------------------------------------
c     advance to next Fourier component.
c-----------------------------------------------------------------------
         expfac=expfac*expfac0
         dexpfac=dexpfac*dexpfac0
      enddo
c-----------------------------------------------------------------------
c     fit Fourier coefficients to cubic splines as functions of x.
c-----------------------------------------------------------------------
      fst%cs%xs=fst%xs
c      fst%cs%fs=RESHAPE(coef,(/mx+1,(mband+1)*nqty/))
c     following line is replace of RESHAPE.
      j = 0
      do iq = 1, nqty
          do m = 0, mband
              j = j + 1
              fst%cs%fs(:, j) = coef(:, m, iq)
          enddo
      enddo
      if(fit_flag)call cspline_fit(fst%cs,endmode)
      fst%cs%name=fst%name
      fst%cs%title(0)="  x   "
      j=0
      do iq=1,nqty
         do m=0,mband
            j=j+1
            write(fst%cs%title(j),'("cs",i1,"_",i2.2)')iq,m
         enddo
      enddo
c-----------------------------------------------------------------------
c     deallocate.
c-----------------------------------------------------------------------
      deallocate (delta,d4fac,delta4)
      deallocate (fs,fsy)
      deallocate (expfac0,expfac)
      deallocate (dexpfac0,dexpfac)
      deallocate (alpha1,alpha2,beta1,beta2)
      deallocate (coef)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_fit_1
c-----------------------------------------------------------------------
c     subprogram 4. fspline_fit_2.
c     fits functions to fsplines using Fast Fourier Transform.
c-----------------------------------------------------------------------
      subroutine fspline_fit_2(fst,endmode,fit_flag)

      type(fspline_type), intent(inout) :: fst
      integer, intent(in) :: endmode
      logical, intent(in) :: fit_flag

      character(64) :: message
      integer :: iqty,j,m,mband,mx,my,nqty,p2
      complex(r8), allocatable, dimension(:,:,:) :: f,g

      complex(r8), allocatable, dimension(:,:) :: temp_in, temp_out
      integer :: ix, k
c-----------------------------------------------------------------------
c     allocate.
c-----------------------------------------------------------------------
      allocate (f(0:fst%my-1,0:fst%mx,fst%nqty),
     $          g(0:fst%my-1,0:fst%mx,fst%nqty))
c-----------------------------------------------------------------------
c     abort if my is not a power of 2.
c-----------------------------------------------------------------------
      p2=1
      my=fst%my
      do
         my=my/2
         if(my == 0)EXIT
         p2=p2*2
      enddo
      if(fst%my /= p2)then
         write(message,'(a,i3,a)')
     $     "fft_fit_2: my = ",fst%my," is not a power of 2"
         call program_stop(message)
      endif
c-----------------------------------------------------------------------
c     abort if 2*mband > my-1.
c-----------------------------------------------------------------------
      if(2*fst%mband > fst%my-1)then
         write(message,'(a,i3,a,i3)')
     $        "fft_fit_2: 2*mband = ",2*fst%mband," > my-1 = ",fst%my-1
         call program_stop(message)
      endif
c-----------------------------------------------------------------------
c     copy sizes and zero Fourier coefficients.
c-----------------------------------------------------------------------
      mx=fst%mx
      my=fst%my
      mband=fst%mband
      nqty=fst%nqty
c-----------------------------------------------------------------------
c     set up for Fast Fourier Transform.
c-----------------------------------------------------------------------


      do iqty=1,nqty
         f(:,:,iqty)=TRANSPOSE(fst%fs(:,0:my-1,iqty))
      enddo

c    RESHAPE made unknown ERROR so replaced.
c    323 ~ 345 line ( from allocate (temp_in ...) to deallocate(temp_in, ))
c    is replace of
c     g=RESHAPE(fft_run(RESHAPE(f,(/my,(mx+1)*nqty/)),-1),
c   $     (/my,mx+1,nqty/))

      allocate (temp_in(my, (mx+1)*nqty))
      k = 0
      do iqty = 1, nqty
          do ix = 0, mx
              k = k + 1
              temp_in(:, k) = f(:, ix, iqty)
          end do
      end do

      allocate (temp_out(my, (mx+1)*nqty))

      temp_out =  fft_run(temp_in, -1)


      k = 0
      do iqty = 1, nqty
          do ix = 0, mx
              k = k + 1
              g(:, ix, iqty) = temp_out(:, k)
          end do
      end do

      deallocate(temp_in, temp_out)
c-----------------------------------------------------------------------
c     copy Fourier coefficients from g to the cspline structure
c-----------------------------------------------------------------------
      j=1
      do iqty=1,nqty
         do m=0,mband
            fst%cs%fs(:,j)=g(m,:,iqty)
            j=j+1
         enddo
      enddo
c-----------------------------------------------------------------------
c     fit Fourier coefficients to cubic splines as functions of x.
c-----------------------------------------------------------------------
      fst%cs%xs=fst%xs
      if(fit_flag)call cspline_fit(fst%cs,endmode)
      fst%cs%name=fst%name
      fst%cs%title(0)="  x   "
      j=0
      do iqty=1,nqty
         do m=0,mband
            j=j+1
            write(fst%cs%title(j),'("cs",i1,"_",i2.2)')iqty,m
         enddo
      enddo
c-----------------------------------------------------------------------
c     dellocate.
c-----------------------------------------------------------------------
      deallocate (f,g)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_fit_2
c-----------------------------------------------------------------------
c     subprogram 5. fspline_eval.
c     evaluates fspline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_eval(fst,x,y,mode)

      type(fspline_type), intent(inout) :: fst
      real(r8), intent(in) :: x,y
      integer, intent(in) :: mode

      integer :: m
      complex(r8) :: expfac,expfac0
      complex(r8), dimension(fst%nqty) :: term,termx,termxx
      complex(r8), dimension(0:fst%mband,fst%nqty) :: c,cx,cxx
c-----------------------------------------------------------------------
c     evaluate cubic splines and m = 0 terms for functions.
c-----------------------------------------------------------------------
      call cspline_eval(fst%cs,x,mode)
      c=RESHAPE(fst%cs%f,(/fst%mband+1,fst%nqty/))
      fst%f=c(0,:)
c-----------------------------------------------------------------------
c     evaluate m = 0 terms for first derivatives.
c-----------------------------------------------------------------------
      if(mode > 0)then
         cx=RESHAPE(fst%cs%f1,(/fst%mband+1,fst%nqty/))
         fst%fx=cx(0,:)
         fst%fy=0
      endif
c-----------------------------------------------------------------------
c     evaluate m = 0 terms for second derivatives.
c-----------------------------------------------------------------------
      if(mode > 1)then
         cxx=RESHAPE(fst%cs%f2,(/fst%mband+1,fst%nqty/))
         fst%fxx=cxx(0,:)
         fst%fyy=0
         fst%fxy=0
      endif
c-----------------------------------------------------------------------
c     evaluate m > 1 for functions.
c-----------------------------------------------------------------------
      expfac0=EXP(ifac*y)
      expfac=2
      do m=1,fst%mband
         expfac=expfac*expfac0
         term=c(m,:)*expfac
         fst%f=fst%f+term
         if(mode < 1)cycle
c-----------------------------------------------------------------------
c     evaluate m > 1 for first derivatives.
c-----------------------------------------------------------------------
         termx=cx(m,:)*expfac
         fst%fx=fst%fx+termx
         fst%fy=fst%fy+term*ifac*m
         if(mode < 2)cycle
c-----------------------------------------------------------------------
c     evaluate m > 1 for second derivatives.
c-----------------------------------------------------------------------
         termxx=cxx(m,:)*expfac
         fst%fxx=fst%fxx+termxx
         fst%fxy=fst%fxy+termx*ifac*m
         fst%fyy=fst%fyy-term*m*m
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_eval
c-----------------------------------------------------------------------
c     subprogram 7. fspline_eval_external.
c     evaluates fspline function (parallel), preserving original logic.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_eval_external(fst, x, y, mode, f_f,
     $                       f_fx, f_fy, f_fxx, f_fxy, f_fyy)

      type(fspline_type), intent(in) :: fst
      real(r8), intent(in) :: x, y
      integer, intent(in) :: mode

      integer :: m
      complex(r8) :: expfac, expfac0
      complex(r8), dimension(fst%nqty) :: term, termx, termxx
      complex(r8), dimension(0:fst%mband, fst%nqty) :: c, cx, cxx

      ! Local arrays
      complex(r8), dimension(fst%cs%nqty) :: cs_f, cs_f1, cs_f2

      ! Output arrays
      real(r8), dimension(fst%nqty), intent(out) :: f_f
      real(r8), dimension(fst%nqty), intent(out), optional :: f_fx,
     $  f_fy
      real(r8), dimension(fst%nqty), intent(out), optional :: f_fxx,
     $  f_fxy, f_fyy

c-----------------------------------------------------------------------
c     zero out output arrays
c-----------------------------------------------------------------------
      f_f = 0.0_r8
      if (present(f_fx)) f_fx = 0.0_r8
      if (present(f_fy)) f_fy = 0.0_r8
      if (present(f_fxx)) f_fxx = 0.0_r8
      if (present(f_fxy)) f_fxy = 0.0_r8
      if (present(f_fyy)) f_fyy = 0.0_r8

c-----------------------------------------------------------------------
c     evaluate cubic splines
c-----------------------------------------------------------------------
      select case(mode)
      case (0)
          call cspline_eval_external(fst%cs, x,
     $      cs_f)
      case (1)
          call cspline_eval_external(fst%cs, x,
     $      cs_f, cs_f1)
      case (2)
          call cspline_eval_external(fst%cs, x, cs_f,
     $      cs_f1, cs_f2)
      end select

c-----------------------------------------------------------------------
c     evaluate m = 0 terms for functions.
c-----------------------------------------------------------------------
      c = RESHAPE(cs_f, (/fst%mband + 1, fst%nqty/))
      f_f = real(c(0, :))

c-----------------------------------------------------------------------
c     evaluate m = 0 terms for first derivatives.
c-----------------------------------------------------------------------
      if (mode > 0) then
         cx = RESHAPE(cs_f1, (/fst%mband + 1, fst%nqty/))
         f_fx = real(cx(0, :))
         f_fy = 0.0_r8
      endif

c-----------------------------------------------------------------------
c     evaluate m = 0 terms for second derivatives.
c-----------------------------------------------------------------------
      if (mode > 1) then
         cxx = RESHAPE(cs_f2, (/fst%mband + 1, fst%nqty/))
         f_fxx = real(cxx(0, :))
         f_fyy = 0.0_r8
         f_fxy = 0.0_r8
      endif
c-----------------------------------------------------------------------
c     evaluate m > 1 for functions
c-----------------------------------------------------------------------
      expfac0 = EXP(ifac * y)
      expfac = 2.0_r8
      do m = 1, fst%mband
         expfac = expfac * expfac0
         term = c(m, :) * expfac

         f_f = f_f + real(term)
         if (mode < 1) cycle
c-----------------------------------------------------------------------
c     evaluate m > 1 for first derivatives.
c-----------------------------------------------------------------------
         termx = cx(m, :) * expfac
         f_fx = f_fx + real(termx)
         f_fy = f_fy + real(term * ifac * m)
         if (mode < 2) cycle
c-----------------------------------------------------------------------
c     evaluate m > 1 for second derivatives.
c-----------------------------------------------------------------------
         termxx = cxx(m, :) * expfac
         f_fxx = f_fxx + real(termxx)
         f_fxy = f_fxy + real(termx * ifac * m)
         f_fyy = f_fyy - real(term) * m * m
      enddo
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_eval_external
c-----------------------------------------------------------------------
c     subprogram 6. fspline_all_eval.
c     evaluates fsplines in all intervals.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_all_eval(fst,dx,dy,f,fx,fy,fxx,fyy,fxy,mode)

      type(fspline_type), intent(inout) :: fst
      real(r8), intent(in) :: dx,dy
      real(r8), intent(out), dimension(fst%mx,fst%my,fst%nqty) ::
     $     f,fx,fy,fxx,fyy,fxy
      integer, intent(in) :: mode

      integer :: m,iy
      real(r8), dimension(fst%my) :: y
      complex(r8), dimension(fst%my) :: expfac,expfac0
      complex(r8), allocatable, dimension(:,:) :: f0,f1,f2,f3
      complex(r8), allocatable, dimension(:,:,:) ::
     $     term,termx,termxx
      complex(r8), allocatable, dimension(:,:,:) :: c,cx,cxx
c-----------------------------------------------------------------------
c     allocate.
c-----------------------------------------------------------------------
      allocate (f0(fst%my,fst%nqty*(fst%mx+1)),
     $          f1(fst%my,fst%nqty*(fst%mx+1)),
     $          f2(fst%my,fst%nqty*(fst%mx+1)),
     $          f3(fst%my,fst%nqty*(fst%mx+1)))
      allocate (term(fst%mx,fst%my,fst%nqty),
     $          termx(fst%mx,fst%my,fst%nqty),
     $          termxx(fst%mx,fst%my,fst%nqty))
      allocate (c(fst%mx,0:fst%mband,fst%nqty),
     $          cx(fst%mx,0:fst%mband,fst%nqty),
     $          cxx(fst%mx,0:fst%mband,fst%nqty))
c-----------------------------------------------------------------------
c     evaluate cubic splines and m = 0 terms for functions.
c-----------------------------------------------------------------------
      call cspline_all_eval(fst%cs,dx,f0,f1,f2,f3,mode)
      c=RESHAPE(f0,(/fst%mx,fst%mband+1,fst%nqty/))
      do iy=1,fst%my
         f(:,iy,:)=c(:,0,:)
      enddo
c-----------------------------------------------------------------------
c     evaluate m = 0 terms for first derivatives.
c-----------------------------------------------------------------------
      if(mode > 0)then
         cx=RESHAPE(f1,(/fst%mx,fst%mband+1,fst%nqty/))
         do iy=1,fst%my
            fx(:,iy,:)=cx(:,0,:)
         enddo
         fy=0
      endif
c-----------------------------------------------------------------------
c     evaluate m = 0 terms for second derivatives.
c-----------------------------------------------------------------------
      if(mode > 1)then
         cxx=RESHAPE(f2,(/fst%mx,fst%mband+1,fst%nqty/))
         do iy=1,fst%my
            fxx(:,iy,:)=cxx(:,0,:)
         enddo
         fxy=0
         fyy=0
      endif
c-----------------------------------------------------------------------
c     evaluate m > 0 terms for functions.
c-----------------------------------------------------------------------
      y=fst%ys(0:fst%my-1)+dy*(fst%ys(1:fst%my)-fst%ys(0:fst%my-1))
      expfac0=EXP(ifac*y)
      expfac=2
      do m=1,fst%mband
         expfac=expfac*expfac0
         do iy=1,fst%my
            term(:,iy,:)=c(:,m,:)*expfac(iy)
         enddo
         f=f+term
         if(mode < 1)cycle
c-----------------------------------------------------------------------
c     evaluate m > 0 terms for first derivatives.
c-----------------------------------------------------------------------
         do iy=1,fst%my
            termx(:,iy,:)=cx(:,m,:)*expfac(iy)
         enddo
         fx=fx+termx
         fy=fy+term*ifac*m
         if(mode < 2)cycle
c-----------------------------------------------------------------------
c     evaluate m > 0 terms for second derivatives.
c-----------------------------------------------------------------------
         do iy=1,fst%my
            termxx(:,iy,:)=cxx(:,m,:)*expfac(iy)
         enddo
         fxx=fxx+termxx
         fxy=fxy+termx*ifac*m
         fyy=fyy-term*m*m
      enddo
c-----------------------------------------------------------------------
c     deallcoate.
c-----------------------------------------------------------------------
      deallocate (f0,f1,f2,f3)
      deallocate (term,termx,termxx)
      deallocate (c,cx,cxx)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_all_eval
c-----------------------------------------------------------------------
c     subprogram 8. fspline_write_xy.
c     produces ascii and binary output for fspline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_write_xy(fst,out,bin,interpolate,filename)

      type(fspline_type), intent(inout) :: fst
      logical, intent(in) :: out,bin,interpolate
      character(*) :: filename

      integer :: ix,iy,jx,jy,iqty
      real(r8) :: x,y,dx,dy

      character(80) :: format2,format1
c-----------------------------------------------------------------------
c     write formats.
c-----------------------------------------------------------------------
 10   format(/1x,'ix = ',i3,', x = ',1p,e11.3)
 20   format(/1x,'ix = ',i3,', jx = ',i1,', x = ',1p,e11.3)
 30   format('(/4x,"iy",6x,"y",4x,',i1,'(6x,"f",i1,3x)/)')
 40   format('(i6,1p,e11.3,',i3.3,'e11.3)')
c-----------------------------------------------------------------------
c     open binary output file.
c-----------------------------------------------------------------------
      if(.not. (out. OR. bin))return
      if(bin)call bin_open(bin_unit,TRIM(filename),
     $     "UNKNOWN","REWinD","none")
c-----------------------------------------------------------------------
c     write input data.
c-----------------------------------------------------------------------
      if(out)then
         write(out_unit,'(1x,a)')"input data"
         write(format1,30)fst%nqty
         write(format2,40)fst%nqty
      endif
      do iy=0,fst%my
         y=fst%ys(iy)
         if(out)then
            write(out_unit,10)iy,fst%ys(iy)
            write(out_unit,format1)(iqty,iqty=1,fst%nqty)
         endif
         do ix=0,fst%mx
            x=fst%xs(ix)
            fst%f=fst%fs(ix,iy,:)
            if(out)write(out_unit,format2)ix,x,fst%f
            if(bin)write(bin_unit)real(x,4),real(fst%f,4)
         enddo
         if(out)write(out_unit,format1)(iqty,iqty=1,fst%nqty)
         if(bin)write(bin_unit)
      enddo
c-----------------------------------------------------------------------
c     begin loops over y for interpolated data.
c-----------------------------------------------------------------------
      if(interpolate)then
         if(out)write(out_unit,'(1x,a)')"interpolated data"
         do iy=0,fst%my-1
            dy=(fst%ys(iy+1)-fst%ys(iy))/4
            do jy=0,4
               y=fst%ys(iy)+dy*jy
               if(out)then
                  write(out_unit,20)iy,jy,y
                  write(out_unit,format1)(iqty,iqty=1,fst%nqty)
               endif
c-----------------------------------------------------------------------
c     begin loops over x for interpolated data.
c-----------------------------------------------------------------------
               do ix=0,fst%mx-1
                  dx=(fst%xs(ix+1)-fst%xs(ix))/4
                  do jx=0,4
                     x=fst%xs(ix)+dx*jx
                     call fspline_eval(fst,x,y,0)
                     if(out)write(out_unit,format2)ix,x,fst%f
                     if(bin)write(bin_unit)real(x,4),real(fst%f,4)
                  enddo
                  if(out)write(out_unit,'()')
               enddo
c-----------------------------------------------------------------------
c     complete loops over y.
c-----------------------------------------------------------------------
               if(out)write(out_unit,format1)(iqty,iqty=1,fst%nqty)
               if(bin)write(bin_unit)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     close binary output file.
c-----------------------------------------------------------------------
      if(bin)call bin_close(bin_unit)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_write_xy
c-----------------------------------------------------------------------
c     subprogram 9. fspline_write_yx.
c     produces ascii and binary output for fspline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_write_yx(fst,out,bin,interpolate,filename)

      type(fspline_type), intent(inout) :: fst
      logical, intent(in) :: out,bin,interpolate
      character(*) :: filename

      integer :: ix,iy,jx,jy,iqty
      real(r8) :: x,y,dx,dy

      character(80) :: format2,format1
c-----------------------------------------------------------------------
c     write formats.
c-----------------------------------------------------------------------
 10   format(/1x,'ix = ',i3,', x = ',1p,e11.3)
 20   format(/1x,'ix = ',i3,', jx = ',i1,', x = ',1p,e11.3)
 30   format('(/4x,"iy",6x,"y",4x,',i1,'(6x,"f",i1,3x)/)')
 40   format('(i6,1p,e11.3,',i3.3,'e11.3)')
c-----------------------------------------------------------------------
c     open binary output file.
c-----------------------------------------------------------------------
      if(.not. (out. OR. bin))return
      if(bin)call bin_open(bin_unit,TRIM(filename),
     $     "UNKNOWN","REWinD","none")
c-----------------------------------------------------------------------
c     write input data.
c-----------------------------------------------------------------------
      if(out)then
         write(out_unit,'(1x,a)')"input data"
         write(format1,30)fst%nqty
         write(format2,40)fst%nqty
      endif
      do ix=0,fst%mx
         x=fst%xs(ix)
         if(out)then
            write(out_unit,10)ix,fst%xs(ix)
            write(out_unit,format1)(iqty,iqty=1,fst%nqty)
         endif
         do iy=0,fst%my
            y=fst%ys(iy)
            fst%f=fst%fs(ix,iy,:)
            if(out)write(out_unit,format2)iy,y,fst%f
            if(bin)write(bin_unit)real(y,4),real(fst%f,4)
         enddo
         if(out)write(out_unit,format1)(iqty,iqty=1,fst%nqty)
         if(bin)write(bin_unit)
      enddo
c-----------------------------------------------------------------------
c     begin loops over x for interpolated data.
c-----------------------------------------------------------------------
      if(interpolate)then
         if(out)write(out_unit,'(1x,a)')"interpolated data"
         do ix=0,fst%mx-1
            dx=(fst%xs(ix+1)-fst%xs(ix))/4
            do jx=0,4
               x=fst%xs(ix)+dx*jx
               if(out)then
                  write(out_unit,20)ix,jx,x
                  write(out_unit,format1)(iqty,iqty=1,fst%nqty)
               endif
c-----------------------------------------------------------------------
c     begin loops over y for interpolated data.
c-----------------------------------------------------------------------
               do iy=0,fst%my-1
                  dy=(fst%ys(iy+1)-fst%ys(iy))/4
                  do jy=0,4
                     y=fst%ys(iy)+dy*jy
                     call fspline_eval(fst,x,y,0)
                     if(out)write(out_unit,format2)iy,y,fst%f
                     if(bin)write(bin_unit)real(y,4),real(fst%f,4)
                  enddo
                  if(out)write(out_unit,'()')
               enddo
c-----------------------------------------------------------------------
c     complete loops over x.
c-----------------------------------------------------------------------
               if(out)write(out_unit,format1)(iqty,iqty=1,fst%nqty)
               if(bin)write(bin_unit)
            enddo
         enddo
      endif
c-----------------------------------------------------------------------
c     close binary output file.
c-----------------------------------------------------------------------
      if(bin)call bin_close(bin_unit)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_write_yx
c-----------------------------------------------------------------------
c     subprogram 10. fspline_copy.
c     copies one fspline type to another.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine fspline_copy(fst1,fst2)

      type(fspline_type), intent(in) :: fst1
      type(fspline_type), intent(inout) :: fst2
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      if(ASSOCIATED(fst2%xs))call fspline_dealloc(fst2)
      call fspline_alloc(fst2,fst1%mx,fst1%my,fst1%mband,fst1%nqty)
      call cspline_copy(fst1%cs,fst2%cs)
      fst2%xs=fst1%xs
      fst2%ys=fst1%ys
      fst2%fs=fst1%fs
      fst2%name=fst1%name
      fst2%title=fst1%title
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_copy
      end module fspline_mod
