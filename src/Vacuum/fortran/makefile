# === CONFIGURABLE ===
FC ?= gfortran  # You can override this: `make FC=ifort`
FFLAGS ?= -fPIC
LDFLAGS ?= -shared
LIBS ?= -framework Accelerate
RECURSFLAG ?= -frecursive
LEGACYFLAG ?= -std=legacy
ZEROFLAG ?= -finit-local-zero
LIBSUFFIX ?= .dylib

OUTLIB = libvac$(LIBSUFFIX)

F90 = $(FC) $(FFLAGS) $(ZEROFLAG) $(RECURSFLAG) $(LEGACYFLAG)

.f.o:
	$(F90) -c $< -o $@

OBJS = \
	bzio.o \
	vacuum_global.o \
	vacuum_io.o \
	vacuum_ma.o \
	vacuum_math.o \
	vacuum_penn.o \
	vacuum_sprk2.o \
	vacuum_ut.o \
	vacuum_vac.o

# targets

all : $(OUTLIB)

$(OUTLIB): $(OBJS)
	$(FC) $(LDFLAGS) -o $@ $^ $(LIBS)
	cp $(OUTLIB) ../../../deps/

clean:
	rm -f *.o *.mod *.original *.dylib

# dependencies

bzio.o : vacuum_global.o
aleg_new.o : vacuum_global.o
vacuum_diag.o: vacuum_global.o
vacuum_io.o: vacuum_global.o
vacuum_ma.o: vacuum_global.o
vacuum_math.o: vacuum_global.o
vacuum_sprk2.o: vacuum_global.o
vacuum_ut.o: vacuum_global.o
vacuum_vac.o: vacuum_global.o
