
# Override FSDB_READER_DIR or VERDI_HOME from the environment if a site wants to
# use an externally installed Verdi instead.
LOCAL_FSDB_READER_DIR := $(CURDIR)/third_party/verdi/FsdbReader
ifeq ($(origin FSDB_READER_DIR), undefined)
  ifneq ($(wildcard $(LOCAL_FSDB_READER_DIR)/ffrAPI.h),)
    FSDB_READER_DIR := $(LOCAL_FSDB_READER_DIR)
  else
    VERDI_HOME ?= /home/yu.zhang/workspace/tools/verdi/verdi/R-2020.12-SP2
    FSDB_READER_DIR := $(VERDI_HOME)/share/FsdbReader
  endif
endif
FSDB_READER_PLATFORM ?= linux64

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS += -I$(FSDB_READER_DIR) \
            -DPPORT_LINUX -DPORT_LINUXAMD64 -DSVR4 \
            -D_GLIBCXX_USE_CXX11_ABI=0 -DNOVAS_OEM -D_REENTRANT \
            -DFSDB_LIBRARY -DFSDBCOPYRIGHT -DVERDISIMDBLICKEY

FSDB_LIB_DIR := $(FSDB_READER_DIR)/$(FSDB_READER_PLATFORM)
LDFLAGS += -Wl,-rpath,$(FSDB_LIB_DIR)
LDLIBS += $(FSDB_LIB_DIR)/libnffr.so $(FSDB_LIB_DIR)/libnsys.so -lpthread -ldl -lm -lrt -lz

BUILD_DIR := build
TOG_TARGET := $(BUILD_DIR)/fsdb_to_tog_cache
POWER_TARGET := $(BUILD_DIR)/fsdb_to_power_label
TOG_SRC := src/fsdb_to_tog_cache.cpp
POWER_SRC := src/fsdb_to_power_label.cpp

.PHONY: all clean fsdb_to_tog_cache fsdb_to_power_label

all: $(TOG_TARGET) $(POWER_TARGET)

fsdb_to_tog_cache: $(TOG_TARGET)

fsdb_to_power_label: $(POWER_TARGET)

$(TOG_TARGET): $(TOG_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

$(POWER_TARGET): $(POWER_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
