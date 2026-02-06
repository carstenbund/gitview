# ADO-SQL Wrapper Library

Modern SQL-backed wrapper for the legacy ADO library (adolib), providing backward-compatible C API with SQL database backend.

## Overview

This library wraps the legacy ADO (adolib) buffer-based interface with a modern SQL backend while maintaining 100% API compatibility with existing C applications. It uses a hybrid C/C++ architecture where C provides the stable API surface and C++ provides type-safe SQL operations.

## Architecture

```
┌─────────────────────────────────────┐
│   Legacy C Applications             │
│   (existing adolib users)           │
└──────────────┬──────────────────────┘
               │ C Buffer Interface (unchanged)
┌──────────────▼──────────────────────┐
│   adolib.c (C API Layer)            │
│   - Buffer management               │
│   - C API compatibility             │
└──────────────┬──────────────────────┘
               │ Internal C/C++ Interface
┌──────────────▼──────────────────────┐
│   SQL Wrapper (C++)                 │
│   - SqlConnection class             │
│   - SqlQuery class                  │
│   - Buffer↔SQL conversion           │
└──────────────┬──────────────────────┘
               │ SQL Interface
┌──────────────▼──────────────────────┐
│   SQL Database                      │
│   (SQLite / PostgreSQL)             │
└─────────────────────────────────────┘
```

## Features

- **Backward Compatible**: Drop-in replacement for existing adolib
- **C API**: Maintains original C buffer interface
- **C++ Implementation**: Type-safe SQL operations with RAII
- **SQL Backend**: SQLite (embedded) or PostgreSQL (server)
- **Schema Management**: Automatic .def file parsing
- **Transaction Support**: ACID guarantees
- **Buffer Conversion**: Automatic translation between buffers and SQL rows

## Building

### Prerequisites

- GCC 7+ or Clang 6+ (C11 support)
- G++ 14+ (C++14 support)
- SQLite 3.x development libraries
- Make

### Ubuntu/Debian
```bash
sudo apt-get install build-essential libsqlite3-dev
```

### CentOS/RHEL
```bash
sudo yum install gcc gcc-c++ sqlite-devel make
```

### Build Commands

```bash
# Build everything (shared + static libraries)
make

# Build with debug symbols
make debug

# Build optimized release
make release

# Run tests (when available)
make test

# Install system-wide (requires sudo)
sudo make install

# Install to custom location
make install PREFIX=/opt/adolib

# Clean build artifacts
make clean
```

## Usage

### C API Example

```c
#include <ado/adolib.h>
#include <stdio.h>

int main() {
    /* Initialize library */
    ado_initialize();

    /* Connect to database */
    ADO_CONNECTION conn = ado_connect("database.db");
    if (!conn) {
        fprintf(stderr, "Failed to connect\n");
        return 1;
    }

    /* Load schema from .def file */
    ado_load_schema("users.def");

    /* Create table */
    ado_create_table(conn, "users");

    /* Create and populate a buffer */
    ADO_BUFFER buf = ado_buffer_create_for_table("users");
    int64_t id = 1;
    char name[] = "John Doe";

    ado_buffer_set_field(buf, "id", &id, sizeof(id));
    ado_buffer_set_field(buf, "name", name, strlen(name));

    /* Insert into database */
    if (ado_insert(conn, "users", buf) == ADO_OK) {
        printf("Record inserted successfully\n");
    }

    /* Select records */
    ADO_BUFFER* results;
    size_t num_results;
    ado_select(conn, "users", "id = 1", &results, &num_results);

    /* Process results */
    for (size_t i = 0; i < num_results; i++) {
        char name_buf[100];
        size_t name_len = sizeof(name_buf);
        ado_buffer_get_field(results[i], "name", name_buf, &name_len);
        printf("Name: %.*s\n", (int)name_len, name_buf);
    }

    /* Cleanup */
    ado_free_results(results, num_results);
    ado_buffer_free(buf);
    ado_disconnect(conn);
    ado_shutdown();

    return 0;
}
```

### C++ API Example

```cpp
#include <ado/sql_wrapper.h>
#include <iostream>

int main() {
    try {
        /* Open database connection */
        ado::SqlConnection conn("database.db");

        /* Create table */
        conn.createTable("users",
            "id INTEGER PRIMARY KEY, "
            "name TEXT, "
            "email TEXT");

        /* Insert with transaction */
        {
            ado::Transaction txn(conn);

            auto query = conn.query()
                .sql("INSERT INTO users (id, name, email) VALUES (?, ?, ?)")
                .bind(1, ado::SqlValue(1))
                .bind(2, ado::SqlValue("John Doe"))
                .bind(3, ado::SqlValue("john@example.com"));

            query.executeUpdate();
            txn.commit();
        }

        /* Select with query builder */
        auto result = conn.query()
            .select("id, name, email")
            .from("users")
            .where("id = ?")
            .bind(1, ado::SqlValue(1))
            .execute();

        /* Process results */
        for (const auto& row : result) {
            std::cout << "Name: " << row["name"].asText() << std::endl;
            std::cout << "Email: " << row["email"].asText() << std::endl;
        }

    } catch (const ado::SqlException& e) {
        std::cerr << "SQL error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### Compiling Applications

```bash
# With C API
gcc -o myapp myapp.c -ladosql -lsqlite3

# With C++ API
g++ -o myapp myapp.cpp -ladosql -lsqlite3

# With custom install location
gcc -o myapp myapp.c -I/opt/adolib/include -L/opt/adolib/lib -ladosql -lsqlite3
```

## Schema Definition Files

Schema files use the `.def` format to define table structures:

```
# users.def - User account table

TABLE users
{
    FIELD id            INT64    PRIMARY_KEY
    FIELD username      STRING   SIZE=50
    FIELD email         STRING   SIZE=255
    FIELD full_name     STRING   SIZE=100  NULLABLE
    FIELD is_active     BOOL     DEFAULT=true
    FIELD balance       DOUBLE   DEFAULT=0.0
    FIELD created_at    INT64
    FIELD updated_at    INT64    NULLABLE
}
```

### Field Types

- `INT8`, `INT16`, `INT32`, `INT64` - Signed integers
- `UINT8`, `UINT16`, `UINT32`, `UINT64` - Unsigned integers
- `FLOAT`, `DOUBLE` - Floating point numbers
- `STRING` - Fixed-size string (requires SIZE modifier)
- `BLOB` - Binary data
- `BOOL` - Boolean (stored as INT8)

### Field Modifiers

- `PRIMARY_KEY` - Mark as primary key
- `NULLABLE` - Allow NULL values
- `SIZE=N` - Required for STRING type
- `DEFAULT=value` - Default value

## Testing

```bash
# Run all tests
make test

# Run specific test
./tests/test_basic
./tests/test_buffer
./tests/test_sql
```

## Directory Structure

```
ado-sql/
├── Makefile                # Build configuration
├── README.md               # This file
├── source/                 # Source code
│   ├── adolib.h           # C API header
│   ├── adolib.c           # C API implementation
│   ├── sql_wrapper.h      # C++ SQL wrapper header
│   ├── sql_wrapper.cpp    # C++ SQL wrapper implementation
│   ├── buffer_conv.h      # Conversion layer header
│   └── buffer_conv.cpp    # Conversion layer implementation
├── build/                  # Build artifacts (generated)
├── lib/                    # Output libraries (generated)
│   ├── libadosql.so       # Shared library
│   └── libadosql.a        # Static library
└── tests/                  # Test programs
    ├── test_basic.c       # Basic functionality tests
    ├── test_buffer.c      # Buffer operations tests
    └── test_sql.cpp       # SQL wrapper tests
```

## Documentation

- [SQL Wrapper Feasibility Analysis](../legacy/docs/sql-wrapper-feasibility.md)
- [Implementation Proposal](../legacy/docs/sql-adolib-implementation-proposal.md)

## Performance Considerations

- Use transactions for batch operations
- Prepare statements are cached automatically
- Connection pooling for multi-threaded applications
- Indexes created automatically for primary keys
- Consider SQLite WAL mode for concurrent access

## Migration Guide

### From Flat Files to SQL

1. Parse schema from .def files: `ado_load_schema("table.def")`
2. Create SQL tables: `ado_create_table(conn, "table_name")`
3. Migrate data using conversion tools (see `database/legacy-data/`)
4. Update connection strings in applications
5. Test with existing application code
6. Performance benchmarking

### Database Backend Selection

**SQLite** (Default):
- Embedded, no server required
- Single file database
- Good for moderate concurrency
- Cross-platform

**PostgreSQL** (Future):
- Client-server architecture
- High concurrency
- Advanced features
- Enterprise deployments

## Troubleshooting

### Compilation Errors

```bash
# Check SQLite installation
pkg-config --libs sqlite3

# Check compiler versions
gcc --version    # Should be 7+
g++ --version    # Should be 14+

# Verify headers
ls -la source/*.h
```

### Runtime Errors

- Check database file permissions
- Verify schema is loaded before operations
- Check SQL error messages: `ado_get_error(conn)`
- Enable debug logging: `make debug`

## License

See LICENSE file in the root directory.

## Contributing

See main repository CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please file an issue in the main repository.
