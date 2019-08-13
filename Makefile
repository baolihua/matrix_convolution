# Makefile for the paho-mqttpp (C++) sample applications

#PAHO_C_LIB_DIR ?= /usr/lib/x86_64-linux-gnu
PAHO_C_LIB_DIR ?= /usr/local/cuda/lib64 
PAHO_C_INC_DIR ?= /usr/local/include

#PAHO_C_LIB_DIR ?= /home/icube/work/pocl/build/lib/CL
#PAHO_C_INC_DIR ?= /home/icube/work/pocl/pocl/include
#TGTS  = async_publish 
TGTS  = test_sobel
all: $(TGTS)

#OPENCV_INC = /usr/local/opencv-3.4/include
OPENCV_INC = /usr/local/include/opencv4
#OPENCV_LIB = /home/lingan/Caffe/install/lib
OPENCV_LIB = /usr/local/lib 


ifneq ($(CROSS_COMPILE),)
  CC  = $(CROSS_COMPILE)gcc
  CXX = $(CROSS_COMPILE)g++
  AR  = $(CROSS_COMPILE)ar
  LD  = $(CROSS_COMPILE)ld
endif

CXXFLAGS += -Wall -std=c++11
CPPFLAGS += -I.. -I$(PAHO_C_INC_DIR) -I$(OPENCV_INC)

#define DEBUG

ifdef DEBUG
  CPPFLAGS += -DDEBUG
  CXXFLAGS += -g -O0
else
  CPPFLAGS += -D_NDEBUG
  CXXFLAGS += -O2
endif

LDLIBS +=-L$(PAHO_C_LIB_DIR) -lOpenCL
LDLIBS_SSL += -L../../lib -L$(PAHO_C_LIB_DIR) -lpaho-mqttpp3 -lpaho-mqtt3as 

LDLIBS += -L$(OPENCV_LIB) -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

test_sobel:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_sobel Image2matrix.cpp main_matrix.cpp  $(LDLIBS)
	#$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_sobel Image2matrix.cpp test_main_matrix.cpp  $(LDLIBS)
	#$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_sobel main_rgb.cpp  $(LDLIBS)
	#$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_sobel test_main_conv.cpp  $(LDLIBS)
# Cleanup

.PHONY: clean distclean

clean:
	rm -f $(TGTS)

distclean: clean

