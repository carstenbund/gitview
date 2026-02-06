# Makefile for ADO-SQL Wrapper Library
# Builds a hybrid C/C++ library with SQL backend support

# ============================================================================
# Configuration
# ============================================================================

CC        = gcc
CXX       = g++
AR        = ar
CFLAGS    = -Wall -Wextra -std=c11 -fPIC -Isource
CXXFLAGS  = -Wall -Wextra -std=c++14 -fPIC -Isource
LDFLAGS   = -shared
LIBS      = -lsqlite3

# Directories
SRC_DIR   = source
BUILD_DIR = build
LIB_DIR   = lib
TEST_DIR  = tests

# Output files
TARGET_SHARED  = $(LIB_DIR)/libadosql.so
TARGET_STATIC  = $(LIB_DIR)/libadosql.a

# ============================================================================
# Source Files
# ============================================================================

# C source files
C_SOURCES = $(SRC_DIR)/adolib.c

# C++ source files
CXX_SOURCES = $(SRC_DIR)/sql_wrapper.cpp \
              $(SRC_DIR)/buffer_conv.cpp

# Object files
C_OBJECTS   = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
CXX_OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
ALL_OBJECTS = $(C_OBJECTS) $(CXX_OBJECTS)

# Header files (for dependency tracking)
HEADERS = $(SRC_DIR)/adolib.h \
          $(SRC_DIR)/sql_wrapper.h \
          $(SRC_DIR)/buffer_conv.h

# ============================================================================
# Build Targets
# ============================================================================

.PHONY: all clean test install uninstall help dirs

# Default target
all: dirs $(TARGET_SHARED) $(TARGET_STATIC)

# Create necessary directories
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(TEST_DIR)

# Shared library
$(TARGET_SHARED): $(ALL_OBJECTS) | $(LIB_DIR)
	@echo "Linking shared library: $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built: $@"

# Static library
$(TARGET_STATIC): $(ALL_OBJECTS) | $(LIB_DIR)
	@echo "Creating static library: $@"
	$(AR) rcs $@ $^
	@echo "Built: $@"

# ============================================================================
# Compilation Rules
# ============================================================================

# Compile C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================================
# Installation
# ============================================================================

PREFIX ?= /usr/local
INSTALL_LIB_DIR = $(PREFIX)/lib
INSTALL_INC_DIR = $(PREFIX)/include/ado

install: all
	@echo "Installing libadosql..."
	install -d $(INSTALL_LIB_DIR)
	install -d $(INSTALL_INC_DIR)
	install -m 0755 $(TARGET_SHARED) $(INSTALL_LIB_DIR)
	install -m 0644 $(TARGET_STATIC) $(INSTALL_LIB_DIR)
	install -m 0644 $(SRC_DIR)/adolib.h $(INSTALL_INC_DIR)
	install -m 0644 $(SRC_DIR)/sql_wrapper.h $(INSTALL_INC_DIR)
	install -m 0644 $(SRC_DIR)/buffer_conv.h $(INSTALL_INC_DIR)
	ldconfig 2>/dev/null || true
	@echo "Installation complete!"
	@echo "  Libraries: $(INSTALL_LIB_DIR)"
	@echo "  Headers:   $(INSTALL_INC_DIR)"

uninstall:
	@echo "Uninstalling libadosql..."
	rm -f $(INSTALL_LIB_DIR)/libadosql.so
	rm -f $(INSTALL_LIB_DIR)/libadosql.a
	rm -rf $(INSTALL_INC_DIR)
	ldconfig 2>/dev/null || true
	@echo "Uninstallation complete!"

# ============================================================================
# Testing
# ============================================================================

# Test programs (to be implemented)
TEST_PROGS = $(TEST_DIR)/test_basic \
             $(TEST_DIR)/test_buffer \
             $(TEST_DIR)/test_sql

test: all $(TEST_PROGS)
	@echo "Running tests..."
	@for test in $(TEST_PROGS); do \
		if [ -f $$test ]; then \
			echo "Running $$test..."; \
			$$test || exit 1; \
		fi \
	done
	@echo "All tests passed!"

# Example test compilation (when test sources exist)
$(TEST_DIR)/test_basic: $(TEST_DIR)/test_basic.c $(TARGET_STATIC)
	$(CC) $(CFLAGS) -o $@ $< $(TARGET_STATIC) $(LIBS)

$(TEST_DIR)/test_buffer: $(TEST_DIR)/test_buffer.c $(TARGET_STATIC)
	$(CC) $(CFLAGS) -o $@ $< $(TARGET_STATIC) $(LIBS)

$(TEST_DIR)/test_sql: $(TEST_DIR)/test_sql.cpp $(TARGET_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(TARGET_STATIC) $(LIBS)

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(LIB_DIR)
	rm -f $(TEST_DIR)/test_*
	@echo "Clean complete!"

# ============================================================================
# Development Helpers
# ============================================================================

# Show compilation commands
verbose: CFLAGS += -v
verbose: CXXFLAGS += -v
verbose: all

# Debug build
debug: CFLAGS += -g -O0 -DDEBUG
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: all

# Release build with optimizations
release: CFLAGS += -O3 -DNDEBUG
release: CXXFLAGS += -O3 -DNDEBUG
release: all

# Check for common issues
check:
	@echo "Checking for header files..."
	@ls -la $(SRC_DIR)/*.h
	@echo "Checking for source files..."
	@ls -la $(SRC_DIR)/*.c $(SRC_DIR)/*.cpp 2>/dev/null || echo "No source files yet"
	@echo "Checking SQLite..."
	@pkg-config --exists sqlite3 && echo "SQLite3 found" || echo "WARNING: SQLite3 not found"

# Display help
help:
	@echo "ADO-SQL Makefile - Available targets:"
	@echo ""
	@echo "  make                Build both shared and static libraries"
	@echo "  make all            Same as make"
	@echo "  make clean          Remove build artifacts"
	@echo "  make install        Install libraries and headers (may need sudo)"
	@echo "  make uninstall      Uninstall libraries and headers"
	@echo "  make test           Build and run tests"
	@echo "  make debug          Build with debug symbols"
	@echo "  make release        Build with optimizations"
	@echo "  make check          Check dependencies and environment"
	@echo "  make help           Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  PREFIX=$(PREFIX)    Installation prefix"
	@echo "  CC=$(CC)            C compiler"
	@echo "  CXX=$(CXX)          C++ compiler"
	@echo ""
	@echo "Example usage:"
	@echo "  make clean && make debug"
	@echo "  sudo make install PREFIX=/usr"
	@echo "  make test"

# ============================================================================
# Dependencies
# ============================================================================

# Automatic dependency generation
-include $(ALL_OBJECTS:.o=.d)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -MM -MT $(@:.d=.o) $< > $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< > $@
