c-----------------------------------------------------------------------
c     file defs.f
c     local defintions for most computers.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. defs_mod.
c-----------------------------------------------------------------------
      module defs_mod
      implicit none

      integer, parameter ::
     $     r4=selected_real_kind(6,37),
     $     r8=selected_real_kind(15,307),
     $     r16=selected_real_kind(33,4931),
     $     i4=selected_int_kind(9),
     $     i8=selected_int_kind(18)

      real(r8), parameter :: pi = 3.14159265358979323846_r8
      real(r8), parameter :: twopi = 6.28318530717958647692_r8
      real(r8), parameter :: alog10 = 2.3025850929940459_r8
      real(r8), parameter :: rtod = 180.0_r8 / pi
      real(r8), parameter :: zero = 0.0_r8
      real(r8), parameter :: one = 1.0_r8
      real(r8), parameter :: two = 2.0_r8
      real(r8), parameter :: half = 0.5_r8
      real(r8), parameter :: six = 6.0_r8
      integer, parameter :: bin_unit = 10
      integer, parameter :: out_unit = 11

      complex(r8), PARAMETER :: ifac=(0,1)

      contains
c-----------------------------------------------------------------------
c     subprogram 1. timer.
c     handles machine-dependent timing statistics.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     delcarations.
c-----------------------------------------------------------------------
      subroutine timer(mode,unit,op_cpuseconds,op_wallseconds)

      integer, intent(in) :: mode,unit
      real(r4), intent(out), optional :: op_cpuseconds,op_wallseconds

      integer(i8), save :: count_rate, wall_start
      real(r4), save :: start
      real(r4) :: seconds
      integer(i8) :: hrs,mins,secs, wall_seconds, count_max
c-----------------------------------------------------------------------
c     format statements.
c-----------------------------------------------------------------------
 10   format(1x,a,1p,e10.3,a)
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      if(mode == 0)then
         call CPU_TIME(start)
         call SYSTEM_CLOCK(wall_start)
      else
         ! report cpu time
         call CPU_time(seconds)
         seconds=seconds-start
         secs = int(seconds)
         hrs = secs/(60*60)
         mins = (secs-hrs*60*60)/60
         secs = secs-hrs*60*60-mins*60
         if(present(op_cpuseconds))then
            ! simply provide the time to the caller
            op_cpuseconds = seconds
         else
            ! write the time to terminal and file
            if(hrs>0)then
               write(*,'(1x,a,i3,a,i2,a,i2,a)') "Total cpu time = ",
     $            hrs," hours, ",mins," minutes, ",secs," seconds"
            elseif(mins>0)then
               write(*,'(1x,a,i2,a,i2,a)') "Total cpu time = ",
     $            mins," minutes, ",secs," seconds"
            elseif(secs>0)then
               write(*,'(1x,a,i2,a)') "Total cpu time = ",secs,
     $            " seconds"
            endif
            write(unit,10) "Total cpu time = ",seconds," seconds"
         endif
         ! report wall time
         call SYSTEM_CLOCK(wall_seconds, count_rate, count_max)
         seconds=real(wall_seconds-wall_start, r4)/real(count_rate, r4)
         secs = int(seconds)
         hrs = secs/(60*60)
         mins = (secs-hrs*60*60)/60
         secs = secs-hrs*60*60-mins*60
         if(present(op_wallseconds))then
            op_wallseconds = seconds
         else
            if(hrs>0)then
               write(*,'(1x,a,i3,a,i2,a,i2,a)') "Total wall time = ",
     $            hrs," hours, ",mins," minutes, ",secs," seconds"
            elseif(mins>0)then
               write(*,'(1x,a,i2,a,i2,a)') "Total wall time = ",
     $            mins," minutes, ",secs," seconds"
            elseif(secs>0)then
               write(*,'(1x,a,i2,a)') "Total wall time = ",secs,
     $            " seconds"
            endif
            write(unit,10) "Total wall time = ",seconds," seconds"
         endif
      endif
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine timer
c-----------------------------------------------------------------------
c     subprogram 2. bin_open.
c     opens a binary input or output file.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bin_open(unit,name,stat,pos,convert_type)

      character(*), intent(in) :: name,stat,pos,convert_type
      integer, intent(in) :: unit
c-----------------------------------------------------------------------
c     open file.
c-----------------------------------------------------------------------
      select case(convert_type)
      case("none")
         open(UNIT=unit,FILE=name,STATUS=stat,POSITION=pos,
     $        FORM="UNformatTED")
      case("big")
         open(UNIT=unit,FILE=name,STATUS=stat,POSITION=pos,
     $        FORM="UNformatTED",CONVERT="BIG_endIAN")
      case("little")
         open(UNIT=unit,FILE=name,STATUS=stat,POSITION=pos,
     $        FORM="UNformatTED",CONVERT="LITTLE_endIAN")
      case default
         call program_stop
     $        ("Cannot recognize convert_type = "//TRIM(convert_type))
      end select
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bin_open
c-----------------------------------------------------------------------
c     subprogram 3. bin_close.
c     close a binary input or output file.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine bin_close(unit)

      integer, intent(in) :: unit
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      close(UNIT=unit)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine bin_close
c-----------------------------------------------------------------------
c     subprogram 4. ascii_open.
c     opens a ascii input or output file.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine ascii_open(unit,name,stat)

      character(*), intent(in) :: name,stat
      integer, intent(in) :: unit
c-----------------------------------------------------------------------
c     open file.
c-----------------------------------------------------------------------
      open(UNIT=unit,FILE=name,STATUS=stat)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine ascii_open
c-----------------------------------------------------------------------
c     subprogram 5. ascii_close.
c     close a ascii input or output file.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine ascii_close(unit)

      integer, intent(in) :: unit
c-----------------------------------------------------------------------
c     work.
c-----------------------------------------------------------------------
      close(UNIT=unit)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      return
      end subroutine ascii_close
c-----------------------------------------------------------------------
c     subprogram 6. program_stop.
c     terminates program with message, calls timer, closes output file.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      subroutine program_stop(message)

      character(*), intent(in) :: message
c-----------------------------------------------------------------------
c     write completion message.
c-----------------------------------------------------------------------
      call timer(1,out_unit)
      call ascii_close(out_unit)
      write(*,'(1x,2a)') 'PROGRAM stop => ', TRIM(message)
c-----------------------------------------------------------------------
c     write completion message.
c-----------------------------------------------------------------------
      stop
      end subroutine program_stop

      end module defs_mod
