# Makefile for Proxy Lab 
#
# You may modify this file any way you like (except for the handin
# rule). You instructor will type "make" on your specific Makefile to
# build your proxy from sources.

CC = gcc
CFLAGS = -g -Wall
LDFLAGS = -lpthread

CXX = g++
CXXOBJS = lru.cpp.o lru_test.cpp.o

all: proxy lru_test lru_test_cpp

lru_test: lru_test.o lru.o csapp.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

proxy: proxy.o csapp.o lru.o pool.o
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

$(CXXOBJS) : %.cpp.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

lru_test_cpp: lru_test.cpp.o lru.cpp.o
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Creates a tarball in ../proxylab-handin.tar that you can then
# hand in. DO NOT MODIFY THIS!
handin:
	(make clean; cd ..; tar cvf $(USER)-proxylab-handin.tar proxylab-handout --exclude tiny --exclude nop-server.py --exclude proxy --exclude driver.sh --exclude port-for-user.pl --exclude free-port.sh --exclude ".*")

clean:
	rm -f *~ *.o proxy lru_test lru_test_cpp core *.tar *.zip *.gzip *.bzip *.gz

