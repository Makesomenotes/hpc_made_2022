CC=gcc-11

CFLAGS=-fopenmp -c -Wall -g -O3 

all: matpow

matpow: matpow.o mtr_lib.a
	$(CC) -fopenmp matpow.o mtr_lib.a -o matpow.exe

matpow.o: matpow.c
	$(CC) $(CFLAGS) matpow.c

mtr_init.o: mtr_init.c
	$(CC) $(CFLAGS) mtr_init.c

mtr_calc.o: mtr_calc.c
	$(CC) $(CFLAGS) mtr_calc.c

mtr_lib.a: mtr_init.o mtr_calc.o
	ar rcs mtr_lib.a mtr_init.o mtr_calc.o

clean:
	rm -rf *.o *.a matpow.exe