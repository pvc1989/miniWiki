# Points to the root of this project, relative to where this file is.
# Remember to tweak this if you move this file.
PROJECT_DIR = .

# Points to the location of libraries
LIB_DIR = .

TESTS = test_math_o.exe test_math_so.exe test_math_a.exe
LIBS = libmath.so libmath.a
OBJS = test_math.o math.o

.PHONY: all clean
	
all: $(LIBS) $(TESTS)
	
clean:
	rm -f $(OBJS) $(LIBS) $(TESTS)

test_math_o.exe: test_math.o math.o
	$(CC) -o $@ $^

test_math_so.exe: test_math.o libmath.so
	$(CC) -o $@ -dynamic $< -L$(LIB_DIR) -lmath

test_math_a.exe: test_math.o libmath.a
	$(CC) -o $@ -static $< -L$(LIB_DIR) -lmath

test_math.o: $(PROJECT_DIR)/test/test_math.c
	$(CC) -c $(CFLAGS) $^ 

math.o: $(PROJECT_DIR)/src/math.c
	$(CC) -c $(CFLAGS) $^

libmath.so: math.o
	$(CC) -shared -fpic -o $@ $^

libmath.a: math.o
	$(AR) $(ARFLAGS) $@ $^
