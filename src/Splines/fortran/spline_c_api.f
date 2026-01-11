c-----------------------------------------------------------------------
c     file spline_c_api.f
c     an interface to the spline libraries for c variables.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. spline_c_api_mod.
c.    1. spline_c_create
c     2. spline_c_destroy
c     3. spline_c_setup
c     4. spline_c_fit
c     5. spline_c_eval
c     6. spline_c_eval_deriv
c     7. spline_c_eval_deriv2
c     8. spline_c_eval_deriv3
c     9. spline_c_int
c     10. cspline_c_create
c     11. cspline_c_destroy
c     12. cspline_c_setup
c     13. cspline_c_fit
c     14. cspline_c_eval
c     15. cspline_c_eval_deriv
c     16. cspline_c_eval_deriv2
c     17. cspline_c_eval_deriv3
c     18. cspline_c_int
c     19. bicube_c_create
c     20. bicube_c_destroy
c     21. bicube_c_setup
c     22. bicube_c_fit
c     23. bicube_c_eval
c     24. bicube_c_eval_deriv
c     25. bicube_c_eval_deriv2
c     26. fspline_c_create
c     27. fspline_c_destroy
c     28. fspline_c_setup
c     29. fspline_c_fit_1
c     30. fspline_c_fit_2
c     31. fspline_c_eval
c     32. fspline_c_eval_deriv
c     33. fspline_c_eval_deriv2
c-----------------------------------------------------------------------
c     subprogram 0. spline_c_api_mod
c     module declarations.
c-----------------------------------------------------------------------
      module spline_c_api_mod
      use iso_c_binding
      use spline_mod
      use cspline_mod
      use bicube_mod
      use fspline_mod
      implicit none
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type, bind(C) :: spline_handle
         type(c_ptr) :: obj
      end type spline_handle

      logical :: debug = .false.

      contains

c-----------------------------------------------------------------------
c     Cubic Spline API
c     This section includes the C API for spline.f.
c-----------------------------------------------------------------------



c-----------------------------------------------------------------------
c     subprogram 1. spline_c_create
c     allocates a spline object
c-----------------------------------------------------------------------
      subroutine spline_c_create(mx, nqty, handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      integer(c_int), value :: mx, nqty
      type(spline_handle), intent(out) :: handle
      type(spline_type), pointer :: spl

      allocate(spl)
      call spline_alloc(spl, mx, nqty)
      handle%obj = c_loc(spl)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_create
c-----------------------------------------------------------------------
c     subprogram 2. spline_c_destroy
c     deallocates a spline object
c-----------------------------------------------------------------------
      subroutine spline_c_destroy(handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(spline_type), pointer :: spl

      call c_f_pointer(handle%obj, spl)
      if (associated(spl)) then
         call spline_dealloc(spl)
         deallocate(spl)
      else
         print *, "spline_c_destroy: handle is not associated "
     $       // "with a valid spline object."
      end if
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_destroy
c-----------------------------------------------------------------------
c     subprogram 3. spline_c_setup
c     sets up the spline object with data.
c-----------------------------------------------------------------------
      subroutine spline_c_setup(handle, xs, fs) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(c_ptr), value :: xs, fs

      type(spline_type), pointer :: spl
      real(c_double), pointer :: x(:), f(:, :)
      integer(i8) :: mx, nqty
      integer(i8) :: i
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_setup: handle is not associated "
     $       // "with a valid spline object."
         return
      end if

      mx = spl%mx
      nqty = spl%nqty

      call c_f_pointer(xs, x, [mx+1])
      call c_f_pointer(fs, f, [mx+1, nqty])

      ! do i = 0, mx
      !    spl%xs(i) = x(i+1)  ! Fortran is 1-based, C is 0-based
      !    spl%fs(i, 1:nqty) = f(i+1, 1:nqty)
      ! end do

      spl%xs = x
      spl%fs = f

      if (debug) then
         print *, "spline_c_setup: setting up spline with "
     $       // "mx = ", mx, " and nqty = ", nqty
         print *, "xs = ", x(1:mx+1)
         print *, "fs = "
     $       // "(", nqty, " quantities):"
         do i = 1, nqty
            print *, "  fs(:,", i, ") = ", f(:, i)
         end do
      end if
c------------------------------------------------------------------------
c     terminate.
c------------------------------------------------------------------------
      return
      end subroutine spline_c_setup
c-----------------------------------------------------------------------
c     subprogram 4. spline_c_fit
c     fits the spline to the data.
c-----------------------------------------------------------------------
      subroutine spline_c_fit(handle, endmode, fs1_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      integer(c_int), value :: endmode
      type(c_ptr), value :: fs1_out

      type(spline_type), pointer :: spl
      real(c_double), pointer :: fs1_out_fort(:,:)
      integer :: mx, nqty
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
            print *, "spline_c_fit: handle is not associated "
     $       // "with a valid spline object."
            return
      end if


      call spline_fit(spl, endmode)


      mx = spl%mx
      nqty = spl%nqty

      call c_f_pointer(fs1_out, fs1_out_fort, [mx+1, nqty])

      fs1_out_fort = spl%fs1

      if (debug) then
            print '(A,*(I0,1X))', "fs1_out_fort dims :",
     $   size(fs1_out_fort,1), size(fs1_out_fort,2)
            print '(A,*(I0,1X))', "spl%fs1 dims      :",
     $  size(spl%fs1,1),       size(spl%fs1,2)
      end if
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_fit
c-----------------------------------------------------------------------
c     subprogram 5. spline_c_eval
c     evaluates the spline at a given point.
c-----------------------------------------------------------------------
      subroutine spline_c_eval(handle, x, f) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f


      real(c_double), pointer :: fi(:)
      type(spline_type), pointer :: spl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_eval: handle is not associated "
     $       // "with a valid spline object."
         return
      end if

      call c_f_pointer(f, fi, [spl%nqty])


      call spline_eval_external(spl, x, fi)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_eval
c-----------------------------------------------------------------------
c     subprogram 6. spline_c_eval_deriv
c     evaluates the spline and its first derivative at a given point.
c-----------------------------------------------------------------------
      subroutine spline_c_eval_deriv(handle, x, f, f1) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1

      real(c_double), pointer :: fi(:), f1i(:)
      type(spline_type), pointer :: spl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_eval: handle is not associated "
     $       // "with a valid spline object."
         return
      end if

      call c_f_pointer(f, fi, [spl%nqty])
      call c_f_pointer(f1, f1i, [spl%nqty])

      call spline_eval_external(spl, x, fi, f1i)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_eval_deriv
c-----------------------------------------------------------------------
c     subprogram 7. spline_c_eval_deriv2
c     evaluates the spline and its first and\
c     second derivatives at a given point.
c-----------------------------------------------------------------------
      subroutine spline_c_eval_deriv2(handle, x, f, f1,
     $     f2) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1, f2

      real(c_double), pointer :: fi(:), f1i(:), f2i(:)
      type(spline_type), pointer :: spl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_eval: handle is not associated "
     $       // "with a valid spline object."
         return
      end if

      call c_f_pointer(f, fi, [spl%nqty])
      call c_f_pointer(f1, f1i, [spl%nqty])
      call c_f_pointer(f2, f2i, [spl%nqty])


      call spline_eval_external(spl, x, fi, f1i, f2i)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_eval_deriv2
c-----------------------------------------------------------------------
c     subprogram 8. spline_c_eval_deriv3
c     evaluates the spline and its first derivative at a given point.
c-----------------------------------------------------------------------
      subroutine spline_c_eval_deriv3(handle, x, f, f1,
     $    f2, f3) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1, f2, f3


      real(c_double), pointer :: fi(:), f1i(:), f2i(:), f3i(:)
      type(spline_type), pointer :: spl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_eval: handle is not associated "
     $       // "with a valid spline object."
         return
      end if

      call c_f_pointer(f, fi, [spl%nqty])
      call c_f_pointer(f1, f1i, [spl%nqty])
      call c_f_pointer(f2, f2i, [spl%nqty])
      call c_f_pointer(f3, f3i, [spl%nqty])


      call spline_eval_external(spl, x, fi, f1i, f2i, f3i)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_eval_deriv3
c-----------------------------------------------------------------------
c     subprogram 9. spline_c_int
c     integrates the spline and returns the results.
c-----------------------------------------------------------------------
      subroutine spline_c_int(handle, fsi_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(c_ptr), value :: fsi_out

      type(spline_type), pointer :: spl
      real(c_double), pointer :: fsi_out_fort(:,:)
      integer :: mx, nqty, i, j
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, spl)
      if (.not. associated(spl)) then
         print *, "spline_c_int: handle is not associated."
         return
      end if

      call spline_int(spl)

      mx = spl%mx
      nqty = spl%nqty

      call c_f_pointer(fsi_out, fsi_out_fort, [mx+1, nqty])

      fsi_out_fort = spl%fsi
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine spline_c_int

c-----------------------------------------------------------------------
c     Complex Cubic Spline API
c     This section includes the C API for cspline.f.
c-----------------------------------------------------------------------


c-----------------------------------------------------------------------
c     subprogram 10. cspline_c_create
c     allocates a spline object
c-----------------------------------------------------------------------
      subroutine cspline_c_create(mx, nqty, handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      integer(c_int), value :: mx, nqty
      type(spline_handle), intent(out) :: handle
      type(cspline_type), pointer :: cspl

      allocate(cspl)
      call cspline_alloc(cspl, mx, nqty)
      handle%obj = c_loc(cspl)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_create
c-----------------------------------------------------------------------
c     subprogram 11. cspline_c_destroy
c     deallocates a spline object
c-----------------------------------------------------------------------
      subroutine cspline_c_destroy(handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(cspline_type), pointer :: cspl

      call c_f_pointer(handle%obj, cspl)
      if (associated(cspl)) then
         call cspline_dealloc(cspl)
         deallocate(cspl)
      else
         print *, "cspline_c_destroy: handle is not associated "
     $       // "with a valid cspline object."
      end if
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_destroy
c-----------------------------------------------------------------------
c     subprogram 12. cspline_c_setup
c     sets up the spline object with data.
c-----------------------------------------------------------------------
      subroutine cspline_c_setup(handle, xs, fs) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(c_ptr), value :: xs, fs

      type(cspline_type), pointer :: cspl
      real(c_double), pointer :: x(:)
      complex(c_double), pointer :: f(:, :)
      integer(i8) :: mx, nqty
      integer(i8) :: i
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_setup: handle is not associated "
     $       // "with a valid cspline object."
         return
      end if

      mx = cspl%mx
      nqty = cspl%nqty

      call c_f_pointer(xs, x, [mx+1])
      call c_f_pointer(fs, f, [mx+1, nqty])

      cspl%xs = x
      cspl%fs = f

      if (debug) then
         print *, "cspline_c_setup: setting up spline with "
     $       // "mx = ", mx, " and nqty = ", nqty
         print *, "xs = ", x(1:mx+1)
         print *, "fs = "
     $       // "(", nqty, " quantities):"
         do i = 1, nqty
            print *, "  fs(:,", i, ") = ", f(:, i)
         end do
      end if
c------------------------------------------------------------------------
c     terminate.
c------------------------------------------------------------------------
      return
      end subroutine cspline_c_setup
c-----------------------------------------------------------------------
c     subprogram 13. cspline_c_fit
c     fits the spline to the data.
c-----------------------------------------------------------------------
      subroutine cspline_c_fit(handle, endmode,fs1_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      integer(c_int), value :: endmode
      type(c_ptr), value :: fs1_out

      type(cspline_type), pointer :: cspl
      complex(c_double), pointer :: fs1_out_fort(:,:)
      integer :: mx, nqty
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------

      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
            print *, "cspline_c_fit: handle is not associated "
     $       // "with a valid cspline object."
            return
      end if

      call cspline_fit(cspl, endmode)

      mx = cspl%mx
      nqty = cspl%nqty

      call c_f_pointer(fs1_out, fs1_out_fort, [mx+1, nqty])

      fs1_out_fort = cspl%fs1
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_fit
c-----------------------------------------------------------------------
c     subprogram 14. cspline_c_eval
c     evaluates the spline at a given point.
c-----------------------------------------------------------------------
      subroutine cspline_c_eval(handle, x, f) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f

      complex(c_double), pointer :: fi(:)
      type(cspline_type), pointer :: cspl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_eval: handle is not associated "
     $       // "with a valid cspline object."
         return
      end if

      call c_f_pointer(f, fi, [cspl%nqty])

      call cspline_eval_external(cspl, x, fi)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_eval
c-----------------------------------------------------------------------
c     subprogram 15. cspline_c_eval_deriv
c     evaluates the spline and its first derivative at a given point.
c-----------------------------------------------------------------------
      subroutine cspline_c_eval_deriv(handle, x, f, f1) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1

      complex(c_double), pointer :: fi(:), f1i(:)
      type(cspline_type), pointer :: cspl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_eval: handle is not associated "
     $       // "with a valid cspline object."
         return
      end if

      call c_f_pointer(f, fi, [cspl%nqty])
      call c_f_pointer(f1, f1i, [cspl%nqty])



      call cspline_eval_external(cspl, x, fi, f1i)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_eval_deriv
c-----------------------------------------------------------------------
c     subprogram 16. cspline_c_eval_deriv2
c     evaluates the spline and its first and\
c     second derivatives at a given point.
c-----------------------------------------------------------------------
      subroutine cspline_c_eval_deriv2(handle, x, f, f1,
     $     f2) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1, f2

      complex(c_double), pointer :: fi(:), f1i(:), f2i(:)
      type(cspline_type), pointer :: cspl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_eval: handle is not associated "
     $       // "with a valid cspline object."
         return
      end if

      call c_f_pointer(f, fi, [cspl%nqty])
      call c_f_pointer(f1, f1i, [cspl%nqty])
      call c_f_pointer(f2, f2i, [cspl%nqty])


      call cspline_eval_external(cspl, x, fi, f1i, f2i)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_eval_deriv2
c-----------------------------------------------------------------------
c     subprogram 17. cspline_c_eval_deriv3
c     evaluates the spline and its first derivative at a given point.
c-----------------------------------------------------------------------
      subroutine cspline_c_eval_deriv3(handle, x, f, f1,
     $    f2, f3) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x
      type(c_ptr), value :: f, f1, f2, f3

      complex(c_double), pointer :: fi(:), f1i(:), f2i(:), f3i(:)
      type(cspline_type), pointer :: cspl
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_eval: handle is not associated "
     $       // "with a valid cspline object."
         return
      end if

      call c_f_pointer(f, fi, [cspl%nqty])
      call c_f_pointer(f1, f1i, [cspl%nqty])
      call c_f_pointer(f2, f2i, [cspl%nqty])
      call c_f_pointer(f3, f3i, [cspl%nqty])

      call cspline_eval_external(cspl, x, fi, f1i, f2i, f3i)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_eval_deriv3
c-----------------------------------------------------------------------
c     subprogram 18. cspline_c_int
c     integrates the complex spline and returns the results.
c-----------------------------------------------------------------------
      subroutine cspline_c_int(handle, fsi_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      use iso_c_binding, only: c_ptr, c_f_pointer, c_double_complex
      type(spline_handle), value :: handle
      type(c_ptr), value :: fsi_out

      type(cspline_type), pointer :: cspl
      complex(c_double_complex), pointer :: fsi_out_fort(:,:)
      integer :: mx, nqty, i, j
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, cspl)
      if (.not. associated(cspl)) then
         print *, "cspline_c_int: handle is not associated."
         return
      end if

      call cspline_int(cspl)

      mx = cspl%mx
      nqty = cspl%nqty

      call c_f_pointer(fsi_out, fsi_out_fort, [mx+1, nqty])

      fsi_out_fort = cspl%fsi
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine cspline_c_int


c-----------------------------------------------------------------------
c     Bicubic Spline API
c     This section includes the C API for bicube.f.
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     subprogram 19. bicube_c_create
c     allocates a bicubic spline object
c-----------------------------------------------------------------------
      subroutine bicube_c_create(mx, my, nqty, handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      integer(c_int), value :: mx, my, nqty
      type(spline_handle), intent(out) :: handle
      type(bicube_type), pointer :: bicube

      allocate(bicube)
      call bicube_alloc(bicube, mx, my, nqty)
      handle%obj = c_loc(bicube)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_create
c-----------------------------------------------------------------------
c     subprogram 20. bicube_c_destroy
c     deallocates a bicubic spline object
c-----------------------------------------------------------------------
      subroutine bicube_c_destroy(handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(bicube_type), pointer :: bicube

      call c_f_pointer(handle%obj, bicube)
      if (associated(bicube)) then
         call bicube_dealloc(bicube)
         deallocate(bicube)
      else
         print *, "bicube_c_destroy: handle is not associated "
     $       // "with a valid bicubic spline object."
      end if
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_destroy
c-----------------------------------------------------------------------
c     subprogram 21. bicube_c_setup
c     sets up the bicubic spline object with data.
c-----------------------------------------------------------------------
      subroutine bicube_c_setup(handle, xs, ys, fs) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(c_ptr), value :: xs, ys, fs

      type(bicube_type), pointer :: bicube
      real(c_double), pointer :: x(:), y(:), f(:, :, :)
      integer(i8) :: mx, my, nqty
      integer(i8) :: i
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, bicube)
      if (.not. associated(bicube)) then
         print *, "bicube_c_setup: handle is not associated "
     $       // "with a valid bicubic spline object."
         return
      end if

      mx = bicube%mx
      my = bicube%my
      nqty = bicube%nqty

      call c_f_pointer(xs, x, [mx+1])
      call c_f_pointer(ys, y, [my+1])
      call c_f_pointer(fs, f, [mx+1, my+1, nqty])

      ! do i = 0, mx
      !    spl%xs(i) = x(i+1)  ! Fortran is 1-based, C is 0-based
      !    spl%fs(i, 1:nqty) = f(i+1, 1:nqty)
      ! end do

      bicube%xs = x
      bicube%ys = y
      bicube%fs = f

      if (debug) then
         print *, "bicube_c_setup: setting up bicubic spline with "
     $       // "mx = ", mx, ", my = ", my, " and nqty = ", nqty
         print *, "xs = ", x(1:mx+1)
         print *, "ys = ", y(1:my+1)
         print *, "fs = "
     $       // "(", nqty, " quantities):"
         do i = 1, nqty
            print *, "  fs(:,:,", i, ") = ", f(:,:,i)
         end do
      end if
c------------------------------------------------------------------------
c     terminate.
c------------------------------------------------------------------------
      return
      end subroutine bicube_c_setup
c-----------------------------------------------------------------------
c     subprogram 22. bicube_c_fit
c     fits the bicubic spline to the data.
c-----------------------------------------------------------------------
      subroutine bicube_c_fit(handle, endmode1, endmode2
     $ , fsx, fsy, fsxy) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      integer(c_int), value :: endmode1, endmode2
      type(c_ptr), value :: fsx,fsy,fsxy

      type(bicube_type), pointer :: bicube
      real(c_double), pointer :: fsx_f(:,:,:)
      real(c_double), pointer :: fsy_f(:,:,:)
      real(c_double), pointer :: fsxy_f(:,:,:)
      integer(i8) :: mx, my, nqty
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, bicube)
      if (.not. associated(bicube)) then
            print *, "bicube_c_fit: handle is not associated "
     $       // "with a valid bicubic spline object."
            return
      end if

      call bicube_fit(bicube, endmode1, endmode2)

      mx = bicube%mx
      my = bicube%my
      nqty = bicube%nqty
      call c_f_pointer(fsx, fsx_f, [mx+1, my+1, nqty])
      call c_f_pointer(fsy, fsy_f, [mx+1, my+1, nqty])
      call c_f_pointer(fsxy, fsxy_f, [mx+1, my+1, nqty])

      if (debug) then
            print *, "Pointer shapes"
            print *, "shape(fsx_f)  =", shape(fsx_f)
            print *, "shape(fsy_f)  =", shape(fsy_f)
            print *, "shape(fsxy_f) =", shape(fsxy_f)

            print *, "== Field shapes in bicube type =="
            print *, "shape(bicube%fsx)  =", shape(bicube%fsx)
            print *, "shape(bicube%fsy)  =", shape(bicube%fsy)
            print *, "shape(bicube%fsxy) =", shape(bicube%fsxy)
      end if
      fsx_f = bicube%fsx
      fsy_f = bicube%fsy
      fsxy_f = bicube%fsxy
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_fit
c-----------------------------------------------------------------------
c     subprogram 23. bicube_c_eval
c     evaluates the bicubic spline at a given point.
c-----------------------------------------------------------------------
      subroutine bicube_c_eval(handle, x, y, f) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f

      real(c_double), pointer :: fi(:)
      real(r8), pointer :: fix(:), fiy(:), fxy(:), fxx(:), fyy(:)
      type(bicube_type), pointer :: bicube
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, bicube)
      if (.not. associated(bicube)) then
         print *, "bicube_c_eval: handle is not associated "
     $       // "with a valid bicubic spline object."
         return
      end if

      call c_f_pointer(f, fi, [bicube%nqty])

      call bicube_eval_external(bicube, x, y, 0, fi)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_eval
c-----------------------------------------------------------------------
c     subprogram 24. bicube_c_eval_deriv
c     evaluates the bicube and its first derivative at a given point.
c-----------------------------------------------------------------------
      subroutine bicube_c_eval_deriv(handle, x, y,
     $                                f, f1x, f1y) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f, f1x, f1y

      real(c_double), pointer :: fi(:), f1xi(:), f1yi(:)
      real(r8), pointer :: fxy(:), fxx(:), fyy(:)
      type(bicube_type), pointer :: bicube
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, bicube)
      if (.not. associated(bicube)) then
         print *, "bicube_c_eval_deriv: handle is not associated "
     $       // "with a valid bicubic spline object."
         return
      end if

      call c_f_pointer(f, fi, [bicube%nqty])
      call c_f_pointer(f1x, f1xi, [bicube%nqty])
      call c_f_pointer(f1y, f1yi, [bicube%nqty])
      allocate(fxy(bicube%nqty), fxx(bicube%nqty), fyy(bicube%nqty))


      call bicube_eval_external(bicube, x, y, 1, fi, f1xi, f1yi)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_eval_deriv
c-----------------------------------------------------------------------
c     subprogram 25. bicube_c_eval_deriv2
c     evaluates the bicube and its derivatives at a given point.
c-----------------------------------------------------------------------
      subroutine bicube_c_eval_deriv2(handle, x, y, f, f1x, f1y,
     $                           f2xx, f2xy, f2yy) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f, f1x, f1y, f2xx, f2xy, f2yy

      real(c_double), pointer :: fi(:), f1xi(:), f1yi(:)
      real(c_double), pointer :: f2xxi(:), f2xyi(:), f2yyi(:)
      type(bicube_type), pointer :: bicube
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, bicube)
      if (.not. associated(bicube)) then
         print *, "bicube_c_eval_deriv2: handle is not associated "
     $       // "with a valid bicubic spline object."
         return
      end if

      call c_f_pointer(f, fi, [bicube%nqty])
      call c_f_pointer(f1x, f1xi, [bicube%nqty])
      call c_f_pointer(f1y, f1yi, [bicube%nqty])
      call c_f_pointer(f2xx, f2xxi, [bicube%nqty])
      call c_f_pointer(f2xy, f2xyi, [bicube%nqty])
      call c_f_pointer(f2yy, f2yyi, [bicube%nqty])


      call bicube_eval_external(bicube, x, y, 2, fi, f1xi, f1yi,
     $                                              f2xxi, f2xyi, f2yyi)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bicube_c_eval_deriv2



c-----------------------------------------------------------------------
c     Fourier Spline API
c     This section includes the C API for fspline.f.
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
c     subprogram 26. fspline_c_create
c     allocates a fourier spline object
c-----------------------------------------------------------------------
      subroutine fspline_c_create(mx, my, mband, nqty, handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      integer(c_int), value :: mx, my, mband, nqty
      type(spline_handle), intent(out) :: handle
      type(fspline_type), pointer :: fst

      allocate(fst)
      call fspline_alloc(fst, mx, my, mband, nqty)
      handle%obj = c_loc(fst)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_create

c-----------------------------------------------------------------------
c     subprogram 27. fspline_c_destroy
c     deallocates a fourier spline object
c-----------------------------------------------------------------------
      subroutine fspline_c_destroy(handle) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(fspline_type), pointer :: fst

      call c_f_pointer(handle%obj, fst)
      if (associated(fst)) then
         call fspline_dealloc(fst)
         deallocate(fst)
      else
         print *, "fspline_c_destroy: handle is not associated "
     $       // "with a valid fourier spline object."
      end if
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_destroy

c-----------------------------------------------------------------------
c     subprogram 28. fspline_c_setup
c     sets up the fourier spline object with data.
c-----------------------------------------------------------------------
      subroutine fspline_c_setup(handle, xs, ys, fs) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      type(c_ptr), value :: xs, ys, fs

      type(fspline_type), pointer :: fst
      real(c_double), pointer :: x_c(:), y_c(:)
      real(c_double), pointer :: f_c(:, :, :)
      integer :: mx, my, nqty
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
         print *, "fspline_c_setup: handle is not associated."
         return
      end if

      mx = fst%mx
      my = fst%my
      nqty = fst%nqty

      call c_f_pointer(xs, x_c, [mx+1])
      call c_f_pointer(ys, y_c, [my+1])
      call c_f_pointer(fs, f_c, [mx+1, my+1, nqty])

      ! notE: The data is COPIED from the C pointers to the Fortran
      ! allocated arrays. This is the safe and correct way to handle memory,
      ! preventing crashes on deallocation.
      fst%xs(0:mx) = x_c(1:mx+1)
      fst%ys(0:my) = y_c(1:my+1)
      fst%fs(0:mx, 0:my, 1:nqty) = f_c(1:mx+1, 1:my+1, 1:nqty)
c------------------------------------------------------------------------
c     terminate.
c------------------------------------------------------------------------
      return
      end subroutine fspline_c_setup

c-----------------------------------------------------------------------
c     subprogram 29. fspline_c_fit_1
c     fits the fourier spline using method 1.
c-----------------------------------------------------------------------
      subroutine fspline_c_fit_1(handle, endmode, fit_flag_c) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      integer(c_int), value :: endmode
      logical(c_bool), value :: fit_flag_c

      type(fspline_type), pointer :: fst
      logical :: fit_flag
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
            print *, "fspline_c_fit_1: handle is not associated."
            return
      end if

      fit_flag = fit_flag_c
      call fspline_fit_1(fst, endmode, fit_flag)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_fit_1

c-----------------------------------------------------------------------
c     subprogram 30. fspline_c_fit_2
c     fits the fourier spline using method 2 (FFT).
c-----------------------------------------------------------------------
      subroutine fspline_c_fit_2(handle, endmode, fit_flag_c) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      integer(c_int), value :: endmode
      logical(c_bool), value :: fit_flag_c

      type(fspline_type), pointer :: fst
      logical :: fit_flag
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
            print *, "fspline_c_fit_2: handle is not associated."
            return
      end if

      fit_flag = fit_flag_c
      call fspline_fit_2(fst, endmode, fit_flag)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_fit_2
c-----------------------------------------------------------------------
c     subprogram 31. fspline_c_eval
c     evaluates the fourier spline at a given point.
c-----------------------------------------------------------------------
      subroutine fspline_c_eval(handle, x, y, f_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f_out


      real(c_double), pointer :: f_f(:)
      type(fspline_type), pointer :: fst
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
         print *, "fspline_c_eval: handle is not associated."
         return
      end if

      call c_f_pointer(f_out, f_f, [fst%nqty])


      call fspline_eval_external(fst, x, y, 0, f_f)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_eval

c-----------------------------------------------------------------------
c     subprogram 32. fspline_c_eval_deriv
c     evaluates the fourier spline and its first derivatives.
c-----------------------------------------------------------------------
      subroutine fspline_c_eval_deriv(handle, x, y, f_out,
     $                                fx_out, fy_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f_out, fx_out, fy_out

      type(fspline_type), pointer :: fst
      real(c_double), pointer :: f_f(:), f_fx(:), f_fy(:)
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
         print *, "fspline_c_eval_deriv: handle is not associated."
         return
      end if

      call c_f_pointer(f_out,  f_f,  [fst%nqty])
      call c_f_pointer(fx_out, f_fx, [fst%nqty])
      call c_f_pointer(fy_out, f_fy, [fst%nqty])


      call fspline_eval_external(fst, x, y, 1, f_f, f_fx, f_fy)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_eval_deriv

c-----------------------------------------------------------------------
c     subprogram 33. fspline_c_eval_deriv2
c     evaluates the fourier spline and its second derivatives.
c-----------------------------------------------------------------------
      subroutine fspline_c_eval_deriv2(handle, x, y, f_out,
     $     fx_out, fy_out, fxx_out, fxy_out, fyy_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value :: handle
      real(c_double), value :: x, y
      type(c_ptr), value :: f_out, fx_out, fy_out
      type(c_ptr), value :: fxx_out, fxy_out, fyy_out

      integer(i4) :: s_ix
      real(c_double), pointer :: f_f(:), f_fx(:), f_fy(:)
      real(c_double), pointer :: f_fxx(:), f_fxy(:), f_fyy(:)
      type(fspline_type), pointer :: fst
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
         print *, "fspline_c_eval_deriv2: handle is not associated."
         return
      end if

      call c_f_pointer(f_out,   f_f,   [fst%nqty])
      call c_f_pointer(fx_out,  f_fx,  [fst%nqty])
      call c_f_pointer(fy_out,  f_fy,  [fst%nqty])
      call c_f_pointer(fxx_out, f_fxx, [fst%nqty])
      call c_f_pointer(fxy_out, f_fxy, [fst%nqty])
      call c_f_pointer(fyy_out, f_fyy, [fst%nqty])

      call fspline_eval_external(fst, x, y, 2, f_f, f_fx, f_fy,
     $                           f_fxx, f_fxy, f_fyy)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_eval_deriv2
c-----------------------------------------------------------------------
c     subprogram 34. fspline_c_get_cspline_handle
c     gets a handle to the internal cspline object.
c-----------------------------------------------------------------------
      subroutine fspline_c_get_cspline_handle(handle,
     $  cs_handle_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value, intent(in) :: handle
      type(spline_handle), intent(out) :: cs_handle_out

      type(fspline_type), pointer :: fst
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------

      call c_f_pointer(handle%obj, fst)

      ! Return the C address of the cspline_type component
      cs_handle_out%obj = c_loc(fst%cs)

c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_get_cspline_handle
c-----------------------------------------------------------------------
c     subprogram 35. fspline_c_get_cspline_fs
c     gets a copy of the fourier coefficients (fs from cspline).
c-----------------------------------------------------------------------
      subroutine fspline_c_get_cspline_fs(handle, fs_out) bind(C)
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      type(spline_handle), value, intent(in) :: handle
      type(c_ptr), value, intent(in) :: fs_out

      type(fspline_type), pointer :: fst
      complex(c_double_complex), pointer :: fs_out_fort(:,:)
      integer :: cs_mx, cs_nqty
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      call c_f_pointer(handle%obj, fst)
      if (.not. associated(fst)) then
         print *, "fspline_c_get_cspline_fs: invalid handle."
         return
      end if

      cs_mx = fst%cs%mx
      cs_nqty = fst%cs%nqty
      call c_f_pointer(fs_out, fs_out_fort, [cs_mx+1, cs_nqty])

      fs_out_fort = fst%cs%fs
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine fspline_c_get_cspline_fs
c-----------------------------------------------------------------------
      end module spline_c_api_mod
