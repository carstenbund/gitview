# SQL-ADO Library Implementation Proposal

## Overview

This document provides detailed implementation specifications for wrapping the legacy ADO library (adolib) with a modern SQL interface while maintaining backward compatibility with existing C applications.

## Architecture Design

### Component Layers

#### Layer 1: C Buffer Interface (adolib.h / adolib.c)
**Purpose**: Maintain 100% API compatibility with existing applications

```c
/* adolib.h - Public C API */
#ifndef ADOLIB_H
#define ADOLIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Buffer handle - opaque to users */
typedef struct ado_buffer* ADO_BUFFER;

/* Connection handle */
typedef struct ado_connection* ADO_CONNECTION;

/* Error codes */
typedef enum {
    ADO_OK = 0,
    ADO_ERROR = -1,
    ADO_NOT_FOUND = -2,
    ADO_INVALID_BUFFER = -3,
    ADO_INVALID_CONNECTION = -4,
    ADO_SQL_ERROR = -5
} ado_status_t;

/* Connection management */
ADO_CONNECTION ado_connect(const char* database_path);
void ado_disconnect(ADO_CONNECTION conn);
ado_status_t ado_begin_transaction(ADO_CONNECTION conn);
ado_status_t ado_commit(ADO_CONNECTION conn);
ado_status_t ado_rollback(ADO_CONNECTION conn);

/* Buffer operations */
ADO_BUFFER ado_buffer_create(size_t size);
void ado_buffer_free(ADO_BUFFER buf);
ado_status_t ado_buffer_set_field(ADO_BUFFER buf, const char* field_name,
                                   const void* data, size_t data_len);
ado_status_t ado_buffer_get_field(ADO_BUFFER buf, const char* field_name,
                                   void* data, size_t* data_len);

/* CRUD operations */
ado_status_t ado_insert(ADO_CONNECTION conn, const char* table_name,
                        ADO_BUFFER buf);
ado_status_t ado_update(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause, ADO_BUFFER buf);
ado_status_t ado_delete(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause);
ado_status_t ado_select(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause, ADO_BUFFER* results,
                        size_t* num_results);

/* Error handling */
const char* ado_get_error(ADO_CONNECTION conn);
void ado_clear_error(ADO_CONNECTION conn);

#ifdef __cplusplus
}
#endif

#endif /* ADOLIB_H */
```

#### Layer 2: C++ SQL Wrapper (sql_wrapper.h / sql_wrapper.cpp)
**Purpose**: Provide type-safe SQL operations with RAII and exceptions

```cpp
/* sql_wrapper.h - C++ SQL interface */
#ifndef SQL_WRAPPER_H
#define SQL_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>

namespace ado {

// Forward declarations
class SqlConnection;
class SqlQuery;
class SqlResult;

/* SQL Exception hierarchy */
class SqlException : public std::runtime_error {
public:
    explicit SqlException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class ConnectionError : public SqlException {
public:
    explicit ConnectionError(const std::string& msg)
        : SqlException("Connection error: " + msg) {}
};

class QueryError : public SqlException {
public:
    explicit QueryError(const std::string& msg)
        : SqlException("Query error: " + msg) {}
};

/* SQL Value variant type */
class SqlValue {
public:
    enum Type { NULL_TYPE, INTEGER, REAL, TEXT, BLOB };

    SqlValue();
    SqlValue(int64_t val);
    SqlValue(double val);
    SqlValue(const std::string& val);
    SqlValue(const void* data, size_t len);

    Type type() const { return type_; }
    int64_t asInt() const;
    double asReal() const;
    std::string asText() const;
    std::vector<uint8_t> asBlob() const;

private:
    Type type_;
    int64_t int_val_;
    double real_val_;
    std::string text_val_;
    std::vector<uint8_t> blob_val_;
};

/* SQL Row - named fields */
class SqlRow {
public:
    void addField(const std::string& name, const SqlValue& value);
    SqlValue getField(const std::string& name) const;
    bool hasField(const std::string& name) const;
    std::vector<std::string> fieldNames() const;

private:
    std::map<std::string, SqlValue> fields_;
};

/* SQL Result set */
class SqlResult {
public:
    size_t rowCount() const { return rows_.size(); }
    const SqlRow& getRow(size_t index) const;
    bool empty() const { return rows_.empty(); }

    // Iterator support
    using iterator = std::vector<SqlRow>::const_iterator;
    iterator begin() const { return rows_.begin(); }
    iterator end() const { return rows_.end(); }

private:
    friend class SqlQuery;
    std::vector<SqlRow> rows_;
};

/* SQL Query builder and executor */
class SqlQuery {
public:
    explicit SqlQuery(SqlConnection& conn);
    ~SqlQuery();

    // Query building
    SqlQuery& select(const std::string& columns);
    SqlQuery& from(const std::string& table);
    SqlQuery& where(const std::string& condition);
    SqlQuery& orderBy(const std::string& columns);
    SqlQuery& limit(int count);

    // Parameter binding
    SqlQuery& bind(const std::string& param, const SqlValue& value);
    SqlQuery& bind(int index, const SqlValue& value);

    // Execution
    SqlResult execute();
    int64_t executeUpdate();  // INSERT/UPDATE/DELETE
    int64_t lastInsertId();

    // Raw SQL
    SqlQuery& sql(const std::string& sql_statement);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/* SQL Connection with transaction support */
class SqlConnection {
public:
    explicit SqlConnection(const std::string& database_path);
    ~SqlConnection();

    // Non-copyable
    SqlConnection(const SqlConnection&) = delete;
    SqlConnection& operator=(const SqlConnection&) = delete;

    // Movable
    SqlConnection(SqlConnection&&) noexcept;
    SqlConnection& operator=(SqlConnection&&) noexcept;

    // Connection state
    bool isOpen() const;
    void close();

    // Transaction management
    void beginTransaction();
    void commit();
    void rollback();

    // Query execution
    SqlQuery query();
    int executeRaw(const std::string& sql);

    // Schema operations
    void createTable(const std::string& table_name,
                     const std::string& schema);
    bool tableExists(const std::string& table_name);

    // Error handling
    std::string lastError() const;

private:
    friend class SqlQuery;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/* RAII Transaction guard */
class Transaction {
public:
    explicit Transaction(SqlConnection& conn);
    ~Transaction();

    void commit();
    void rollback();

private:
    SqlConnection& conn_;
    bool committed_;
    bool rolled_back_;
};

} // namespace ado

#endif /* SQL_WRAPPER_H */
```

#### Layer 3: Buffer Conversion (buffer_conv.h / buffer_conv.cpp)
**Purpose**: Convert between C buffers and SQL rows

```cpp
/* buffer_conv.h - Buffer/SQL conversion layer */
#ifndef BUFFER_CONV_H
#define BUFFER_CONV_H

#include "adolib.h"
#include "sql_wrapper.h"
#include <map>
#include <string>

namespace ado {
namespace conversion {

/* Buffer metadata - describes structure */
struct BufferSchema {
    struct Field {
        std::string name;
        enum { INT32, INT64, DOUBLE, STRING, BLOB } type;
        size_t offset;
        size_t size;
    };

    std::string table_name;
    std::vector<Field> fields;
    size_t total_size;
};

/* Schema registry - maps table names to buffer schemas */
class SchemaRegistry {
public:
    static SchemaRegistry& instance();

    void registerSchema(const std::string& table_name,
                       const BufferSchema& schema);
    const BufferSchema& getSchema(const std::string& table_name) const;
    bool hasSchema(const std::string& table_name) const;

private:
    SchemaRegistry() = default;
    std::map<std::string, BufferSchema> schemas_;
};

/* Buffer to SQL conversion */
SqlRow bufferToRow(ADO_BUFFER buf, const std::string& table_name);
SqlValue bufferGetField(ADO_BUFFER buf, const std::string& field_name,
                        const BufferSchema::Field& field_info);

/* SQL to buffer conversion */
ADO_BUFFER rowToBuffer(const SqlRow& row, const std::string& table_name);
void bufferSetField(ADO_BUFFER buf, const std::string& field_name,
                    const SqlValue& value,
                    const BufferSchema::Field& field_info);

/* Schema generation from .def files */
BufferSchema parseDefFile(const std::string& def_file_path);
std::string generateCreateTableSQL(const BufferSchema& schema);

} // namespace conversion
} // namespace ado

#endif /* BUFFER_CONV_H */
```

## Implementation Details

### adolib.c Implementation

```c
/* adolib.c - C API implementation */
#include "adolib.h"
#include "buffer_conv.h"
#include "sql_wrapper.h"
#include <stdlib.h>
#include <string.h>

/* Internal buffer structure */
struct ado_buffer {
    void* data;
    size_t size;
    char* table_name;
};

/* Internal connection structure */
struct ado_connection {
    void* cpp_connection;  /* SqlConnection* */
    char last_error[256];
};

ADO_CONNECTION ado_connect(const char* database_path) {
    ADO_CONNECTION conn = malloc(sizeof(struct ado_connection));
    if (!conn) return NULL;

    /* Create C++ SqlConnection object */
    try {
        ado::SqlConnection* cpp_conn = new ado::SqlConnection(database_path);
        conn->cpp_connection = cpp_conn;
        conn->last_error[0] = '\0';
        return conn;
    } catch (const std::exception& e) {
        strncpy(conn->last_error, e.what(), sizeof(conn->last_error) - 1);
        free(conn);
        return NULL;
    }
}

void ado_disconnect(ADO_CONNECTION conn) {
    if (!conn) return;
    if (conn->cpp_connection) {
        ado::SqlConnection* cpp_conn =
            static_cast<ado::SqlConnection*>(conn->cpp_connection);
        delete cpp_conn;
    }
    free(conn);
}

/* More implementations... */
```

### Build System (Makefile)

```makefile
# Makefile for ADO SQL Wrapper

CXX = g++
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -fPIC
CXXFLAGS = -Wall -Wextra -std=c++14 -fPIC
LDFLAGS = -shared
LIBS = -lsqlite3

# Directories
SRC_DIR = source
BUILD_DIR = build
LIB_DIR = lib

# Source files
C_SOURCES = $(SRC_DIR)/adolib.c
CXX_SOURCES = $(SRC_DIR)/sql_wrapper.cpp $(SRC_DIR)/buffer_conv.cpp

# Object files
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
CXX_OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Targets
TARGET_LIB = $(LIB_DIR)/libadosql.so
TARGET_STATIC = $(LIB_DIR)/libadosql.a

.PHONY: all clean test

all: $(TARGET_LIB) $(TARGET_STATIC)

# Shared library
$(TARGET_LIB): $(C_OBJECTS) $(CXX_OBJECTS) | $(LIB_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Static library
$(TARGET_STATIC): $(C_OBJECTS) $(CXX_OBJECTS) | $(LIB_DIR)
	ar rcs $@ $^

# C compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# C++ compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# Tests
test: all
	$(MAKE) -C tests run
```

## Schema Definition File Format

### Example .def File

```
# customer.def - Customer table definition
TABLE customer
{
    FIELD customer_id    INT64    PRIMARY_KEY
    FIELD name           STRING   SIZE=100
    FIELD email          STRING   SIZE=255
    FIELD balance        DOUBLE
    FIELD created_at     INT64    # Unix timestamp
    FIELD status         INT32
}
```

### Parser Implementation

```cpp
/* .def file parser (buffer_conv.cpp) */
BufferSchema parseDefFile(const std::string& def_file_path) {
    BufferSchema schema;
    std::ifstream file(def_file_path);
    std::string line;
    size_t current_offset = 0;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Parse TABLE line
        if (line.find("TABLE") == 0) {
            size_t pos = line.find_first_of(" \t");
            schema.table_name = line.substr(pos + 1);
            continue;
        }

        // Parse FIELD line
        if (line.find("FIELD") != std::string::npos) {
            BufferSchema::Field field;
            // Parse: FIELD name type [modifiers]
            std::istringstream iss(line);
            std::string keyword, type_str;
            iss >> keyword >> field.name >> type_str;

            // Map type string to enum
            if (type_str == "INT32") {
                field.type = BufferSchema::Field::INT32;
                field.size = 4;
            } else if (type_str == "INT64") {
                field.type = BufferSchema::Field::INT64;
                field.size = 8;
            } else if (type_str == "DOUBLE") {
                field.type = BufferSchema::Field::DOUBLE;
                field.size = 8;
            } else if (type_str == "STRING") {
                field.type = BufferSchema::Field::STRING;
                // Parse SIZE modifier
                std::string modifier;
                while (iss >> modifier) {
                    if (modifier.find("SIZE=") == 0) {
                        field.size = std::stoi(modifier.substr(5));
                    }
                }
            }

            field.offset = current_offset;
            current_offset += field.size;
            schema.fields.push_back(field);
        }
    }

    schema.total_size = current_offset;
    return schema;
}
```

## Testing Strategy

### Unit Tests (Google Test)

```cpp
/* test_buffer_conversion.cpp */
#include <gtest/gtest.h>
#include "adolib.h"
#include "buffer_conv.h"

TEST(BufferConversion, CreateAndConvert) {
    // Setup schema
    BufferSchema schema;
    schema.table_name = "test_table";
    schema.total_size = 16;

    BufferSchema::Field id_field;
    id_field.name = "id";
    id_field.type = BufferSchema::Field::INT64;
    id_field.offset = 0;
    id_field.size = 8;
    schema.fields.push_back(id_field);

    // Register schema
    SchemaRegistry::instance().registerSchema("test_table", schema);

    // Create buffer
    ADO_BUFFER buf = ado_buffer_create(16);
    int64_t id = 12345;
    ado_buffer_set_field(buf, "id", &id, sizeof(id));

    // Convert to SQL row
    SqlRow row = bufferToRow(buf, "test_table");
    EXPECT_EQ(row.getField("id").asInt(), 12345);

    ado_buffer_free(buf);
}
```

### Integration Tests

```c
/* test_integration.c */
#include "adolib.h"
#include <assert.h>
#include <stdio.h>

int main() {
    /* Connect to database */
    ADO_CONNECTION conn = ado_connect("test.db");
    assert(conn != NULL);

    /* Create buffer */
    ADO_BUFFER buf = ado_buffer_create(256);
    int64_t id = 1;
    char name[] = "Test User";

    ado_buffer_set_field(buf, "id", &id, sizeof(id));
    ado_buffer_set_field(buf, "name", name, strlen(name));

    /* Insert */
    ado_status_t status = ado_insert(conn, "users", buf);
    assert(status == ADO_OK);

    /* Select */
    ADO_BUFFER* results;
    size_t num_results;
    status = ado_select(conn, "users", "id = 1", &results, &num_results);
    assert(status == ADO_OK);
    assert(num_results == 1);

    /* Cleanup */
    ado_buffer_free(buf);
    ado_disconnect(conn);

    printf("Integration test passed!\n");
    return 0;
}
```

## Migration Strategy

### Phase 1: Schema Migration
1. Parse all .def files in `database/legacy-data/`
2. Generate CREATE TABLE statements
3. Create SQL database schema
4. Register all buffer schemas in SchemaRegistry

### Phase 2: Data Migration
1. Read legacy data files
2. Parse into buffer format
3. Convert buffers to SQL rows
4. Insert into SQL database
5. Verify data integrity

### Phase 3: Application Integration
1. Link legacy applications with new libadosql
2. Update connection strings to point to SQL database
3. Run application test suites
4. Performance benchmarking
5. Gradual rollout

## Performance Considerations

### Optimization Techniques

1. **Connection Pooling**
   - Reuse database connections
   - Thread-local storage for connections

2. **Prepared Statements**
   - Cache prepared statements
   - Parameterized queries

3. **Batch Operations**
   - Bulk inserts via transactions
   - Batch buffer conversions

4. **Buffer Caching**
   - Cache frequently accessed buffers
   - LRU eviction policy

5. **Indexing Strategy**
   - Index primary keys
   - Index foreign keys
   - Composite indexes for common queries

## Error Handling

### C API Error Handling
```c
ADO_CONNECTION conn = ado_connect("database.db");
if (!conn) {
    fprintf(stderr, "Connection failed\n");
    return -1;
}

ado_status_t status = ado_insert(conn, "users", buf);
if (status != ADO_OK) {
    fprintf(stderr, "Insert failed: %s\n", ado_get_error(conn));
    ado_disconnect(conn);
    return -1;
}
```

### C++ Exception Handling
```cpp
try {
    SqlConnection conn("database.db");
    Transaction txn(conn);

    auto result = conn.query()
        .select("*")
        .from("users")
        .where("id = ?")
        .bind(1, user_id)
        .execute();

    txn.commit();
} catch (const ConnectionError& e) {
    std::cerr << "Connection error: " << e.what() << std::endl;
} catch (const QueryError& e) {
    std::cerr << "Query error: " << e.what() << std::endl;
}
```

## Deployment

### Library Installation
```bash
# Build
make clean && make

# Install headers
cp source/*.h /usr/local/include/ado/

# Install library
cp lib/libadosql.so /usr/local/lib/
ldconfig

# Or install to custom location
make install PREFIX=/opt/adolib
```

### Application Linking
```makefile
# Application Makefile
CFLAGS += -I/usr/local/include/ado
LDFLAGS += -L/usr/local/lib -ladosql -lsqlite3

myapp: myapp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
```

## Documentation

### API Documentation (Doxygen)
```c
/**
 * @brief Create a new ADO buffer
 * @param size Size of the buffer in bytes
 * @return Buffer handle, or NULL on failure
 *
 * The buffer must be freed with ado_buffer_free() when no longer needed.
 */
ADO_BUFFER ado_buffer_create(size_t size);
```

### User Guide Topics
1. Getting Started
2. Buffer Operations
3. CRUD Operations
4. Transaction Management
5. Error Handling
6. Performance Tuning
7. Migration Guide
8. Troubleshooting

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Design & Planning | 1 week | Architecture docs, API specs |
| Core Implementation | 2 weeks | adolib.c, sql_wrapper.cpp, buffer_conv.cpp |
| Schema Parser | 3 days | .def parser, schema registry |
| Testing | 1 week | Unit tests, integration tests |
| Documentation | 3 days | API docs, user guide |
| Performance Tuning | 3 days | Benchmarks, optimization |
| **Total** | **4 weeks** | Complete SQL wrapper library |

## Success Criteria

- [ ] All existing C applications compile without changes
- [ ] All existing C applications pass their test suites
- [ ] SQL queries return correct results matching legacy format
- [ ] Performance within 20% of legacy system
- [ ] No memory leaks (verified with Valgrind)
- [ ] Full test coverage (>80%)
- [ ] Complete documentation
- [ ] Successful migration of sample dataset

## Next Steps

1. Review and approve this proposal
2. Set up development environment
3. Parse first .def file and generate schema
4. Implement minimal viable API (connect, insert, select)
5. Create first integration test
6. Iterate and expand functionality

## Conclusion

This implementation provides a robust, maintainable path forward for modernizing the ADO library while preserving compatibility with existing applications. The layered architecture ensures clean separation of concerns and allows for future enhancements.
