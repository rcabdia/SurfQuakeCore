################## PARAMETERS TO MODIFY *********************************

FC=ifort
FFLAGS = -O3 -heap-arrays

.f90.o:
	$(FC) $(FFLAGS) -c $<

.f.o:
	$(FC) $(FFLAGS) -c $<

all:	gr_xyz
### convms

OBJ1=dimension1.o dimension2.o parameter.o allocateArray.o initdata.o reflect0.o reflect1.o reflect2.o reflect3.o reflect4.o reflect5.o ff0ad.o dll2km.o gr_xyz.o
#######################################
#			AXITRA
#######################################

gr_xyz:	$(INC) $(OBJ1)
	$(FC) $(FFLAGS) -o $@ $(OBJ1) 




######################################
#			CLEAN, TAR, ...
######################################
clean: 
#	del *.o *.mod
	- $(RM) *.o *.mod *.a gr_xyz 2>/dev/null


