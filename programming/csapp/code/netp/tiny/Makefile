CC = gcc
CFLAGS = -O2 -Wall -I ../../include -I .

# This flag includes the Pthreads library on a Linux box.
# Others systems will probably require something different.
LIB = -lpthread

all: tiny cgi

tiny: tiny.c ../../src/csapp.o
	$(CC) $(CFLAGS) -o tiny tiny.c ../../src/csapp.o $(LIB)

cgi:
	(cd cgi-bin; make)

clean:
	rm -f *.o tiny *~
	(cd cgi-bin; make clean)
