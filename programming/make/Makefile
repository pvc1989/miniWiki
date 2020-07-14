# Remember to tweak these values after moving this `Makefile`.
SOURCE_DIR ?= $(PWD)
BUILD_DIR ?= $(PWD)/build

OBJS = test_math.o lib_math.o
LIBS = libmath.so libmath.a
EXES = test_math_o test_math_so test_math_a

.PHONY: all clean
	
all: $(OBJS) $(LIBS) $(EXES)
	
clean:
	cd $(BUILD_DIR); rm -f $(OBJS) $(LIBS) $(EXES)

test_math_o: test_math.o lib_math.o
	cd $(BUILD_DIR); $(CC) -o $@ $^

test_math_so: test_math.o libmath.so
	cd $(BUILD_DIR); $(CC) -o $@ $< -Wl,-rpath,. -L. -lmath

test_math_a: test_math.o libmath.a
	cd $(BUILD_DIR); $(CC) -o $@ -static $< -L. -lmath

test_math.o:
	cd $(BUILD_DIR); $(CC) -o $@ -c $(CFLAGS) $(SOURCE_DIR)/test/math.c

lib_math.o:
	cd $(BUILD_DIR); $(CC) -o $@ -c $(CFLAGS) $(SOURCE_DIR)/src/math.c

libmath.so: lib_math.o
	cd $(BUILD_DIR); $(CC) -o $@ -shared -fpic $(SOURCE_DIR)/src/math.c

libmath.a: lib_math.o
	cd $(BUILD_DIR); $(AR) $(ARFLAGS) $@ $^
