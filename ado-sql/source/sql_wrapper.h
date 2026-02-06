/**
 * @file sql_wrapper.h
 * @brief C++ SQL Wrapper - Type-safe SQL interface with RAII
 *
 * Provides a modern C++ interface for SQL operations with automatic resource
 * management, type safety, and exception-based error handling.
 */

#ifndef SQL_WRAPPER_H
#define SQL_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>
#include <cstdint>

namespace ado {

// Forward declarations
class SqlConnection;
class SqlQuery;
class SqlResult;
class SqlRow;

/* ============================================================================
 * Exception Hierarchy
 * ============================================================================ */

/**
 * @brief Base exception class for SQL errors
 */
class SqlException : public std::runtime_error {
public:
    explicit SqlException(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Connection-related errors
 */
class ConnectionError : public SqlException {
public:
    explicit ConnectionError(const std::string& msg)
        : SqlException("Connection error: " + msg) {}
};

/**
 * @brief Query execution errors
 */
class QueryError : public SqlException {
public:
    explicit QueryError(const std::string& msg)
        : SqlException("Query error: " + msg) {}
};

/**
 * @brief Schema-related errors
 */
class SchemaError : public SqlException {
public:
    explicit SchemaError(const std::string& msg)
        : SqlException("Schema error: " + msg) {}
};

/**
 * @brief Type conversion errors
 */
class TypeError : public SqlException {
public:
    explicit TypeError(const std::string& msg)
        : SqlException("Type error: " + msg) {}
};

/* ============================================================================
 * SQL Value Type
 * ============================================================================ */

/**
 * @brief Variant type for SQL values
 *
 * Represents a single SQL value which can be NULL, INTEGER, REAL, TEXT, or BLOB.
 */
class SqlValue {
public:
    enum Type {
        NULL_TYPE,  /**< SQL NULL */
        INTEGER,    /**< 64-bit integer */
        REAL,       /**< Double precision float */
        TEXT,       /**< UTF-8 string */
        BLOB        /**< Binary data */
    };

    /** Construct NULL value */
    SqlValue();

    /** Construct integer value */
    explicit SqlValue(int32_t val);
    explicit SqlValue(int64_t val);

    /** Construct real value */
    explicit SqlValue(double val);

    /** Construct text value */
    explicit SqlValue(const std::string& val);
    explicit SqlValue(const char* val);

    /** Construct blob value */
    SqlValue(const void* data, size_t len);
    explicit SqlValue(const std::vector<uint8_t>& data);

    /** Get value type */
    Type type() const { return type_; }

    /** Check if value is NULL */
    bool isNull() const { return type_ == NULL_TYPE; }

    /** Convert to integer (throws TypeError if not convertible) */
    int64_t asInt() const;

    /** Convert to real (throws TypeError if not convertible) */
    double asReal() const;

    /** Convert to text (throws TypeError if not convertible) */
    std::string asText() const;

    /** Convert to blob (throws TypeError if not convertible) */
    std::vector<uint8_t> asBlob() const;

    /** Safe conversions with default values */
    int64_t asInt(int64_t default_val) const;
    double asReal(double default_val) const;
    std::string asText(const std::string& default_val) const;

private:
    Type type_;
    int64_t int_val_;
    double real_val_;
    std::string text_val_;
    std::vector<uint8_t> blob_val_;
};

/* ============================================================================
 * SQL Row
 * ============================================================================ */

/**
 * @brief Represents a single row of SQL query results
 *
 * A row is a collection of named fields, each containing a SqlValue.
 */
class SqlRow {
public:
    SqlRow() = default;

    /** Add a field to the row */
    void addField(const std::string& name, const SqlValue& value);

    /** Get field value by name (throws std::out_of_range if not found) */
    const SqlValue& getField(const std::string& name) const;

    /** Get field value by name with default */
    SqlValue getField(const std::string& name, const SqlValue& default_val) const;

    /** Check if field exists */
    bool hasField(const std::string& name) const;

    /** Get all field names */
    std::vector<std::string> fieldNames() const;

    /** Get number of fields */
    size_t fieldCount() const { return fields_.size(); }

    /** Check if row is empty */
    bool empty() const { return fields_.empty(); }

    /** Subscript operator for field access */
    const SqlValue& operator[](const std::string& name) const {
        return getField(name);
    }

private:
    std::map<std::string, SqlValue> fields_;
};

/* ============================================================================
 * SQL Result Set
 * ============================================================================ */

/**
 * @brief Result set from a SELECT query
 *
 * Contains zero or more rows returned by a query.
 */
class SqlResult {
public:
    SqlResult() = default;

    /** Get number of rows */
    size_t rowCount() const { return rows_.size(); }

    /** Get row by index (throws std::out_of_range if invalid) */
    const SqlRow& getRow(size_t index) const;

    /** Get row by index with bounds checking */
    const SqlRow& at(size_t index) const { return getRow(index); }

    /** Check if result is empty */
    bool empty() const { return rows_.empty(); }

    /** Subscript operator for row access */
    const SqlRow& operator[](size_t index) const {
        return rows_[index];
    }

    // Iterator support
    using iterator = std::vector<SqlRow>::const_iterator;
    iterator begin() const { return rows_.begin(); }
    iterator end() const { return rows_.end(); }

private:
    friend class SqlQuery;
    std::vector<SqlRow> rows_;

    void addRow(const SqlRow& row) { rows_.push_back(row); }
};

/* ============================================================================
 * SQL Query Builder
 * ============================================================================ */

/**
 * @brief SQL query builder and executor
 *
 * Provides a fluent interface for building and executing SQL queries.
 */
class SqlQuery {
public:
    explicit SqlQuery(SqlConnection& conn);
    ~SqlQuery();

    // Non-copyable
    SqlQuery(const SqlQuery&) = delete;
    SqlQuery& operator=(const SqlQuery&) = delete;

    // Movable
    SqlQuery(SqlQuery&&) noexcept;
    SqlQuery& operator=(SqlQuery&&) noexcept;

    /* Query building (fluent interface) */
    SqlQuery& select(const std::string& columns);
    SqlQuery& from(const std::string& table);
    SqlQuery& where(const std::string& condition);
    SqlQuery& orderBy(const std::string& columns);
    SqlQuery& limit(int count);
    SqlQuery& offset(int count);

    /* Parameter binding */
    SqlQuery& bind(const std::string& param, const SqlValue& value);
    SqlQuery& bind(int index, const SqlValue& value);
    SqlQuery& bindInt(int index, int64_t value);
    SqlQuery& bindReal(int index, double value);
    SqlQuery& bindText(int index, const std::string& value);
    SqlQuery& bindBlob(int index, const void* data, size_t len);
    SqlQuery& bindNull(int index);

    /* Execution */
    SqlResult execute();              // Execute SELECT query
    int64_t executeUpdate();          // Execute INSERT/UPDATE/DELETE
    void executeNonQuery();           // Execute without returning results

    /* Direct SQL */
    SqlQuery& sql(const std::string& sql_statement);

    /* Get generated SQL (for debugging) */
    std::string getSql() const;

    /* Get last insert row ID */
    int64_t lastInsertId();

    /* Reset query for reuse */
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/* ============================================================================
 * SQL Connection
 * ============================================================================ */

/**
 * @brief Database connection with transaction support
 *
 * Manages a connection to a SQL database (SQLite or PostgreSQL).
 */
class SqlConnection {
public:
    /**
     * @brief Open a database connection
     * @param database_path Path to database file or connection string
     * @throws ConnectionError if connection fails
     */
    explicit SqlConnection(const std::string& database_path);

    /**
     * @brief Close the connection
     */
    ~SqlConnection();

    // Non-copyable
    SqlConnection(const SqlConnection&) = delete;
    SqlConnection& operator=(const SqlConnection&) = delete;

    // Movable
    SqlConnection(SqlConnection&&) noexcept;
    SqlConnection& operator=(SqlConnection&&) noexcept;

    /* Connection state */
    bool isOpen() const;
    void close();

    /* Transaction management */
    void beginTransaction();
    void commit();
    void rollback();
    bool inTransaction() const;

    /* Query creation */
    SqlQuery query();
    SqlQuery query(const std::string& sql);

    /* Direct execution */
    void execute(const std::string& sql);
    SqlResult executeQuery(const std::string& sql);
    int64_t executeUpdate(const std::string& sql);

    /* Schema operations */
    void createTable(const std::string& table_name, const std::string& schema);
    void dropTable(const std::string& table_name);
    bool tableExists(const std::string& table_name);
    std::vector<std::string> listTables();

    /* Error handling */
    std::string lastError() const;

    /* Get last insert row ID */
    int64_t lastInsertId() const;

    /* Database info */
    std::string databasePath() const;

private:
    friend class SqlQuery;
    class Impl;
    std::unique_ptr<Impl> impl_;

    void* getNativeHandle();  // For internal use by SqlQuery
};

/* ============================================================================
 * Transaction RAII Guard
 * ============================================================================ */

/**
 * @brief RAII transaction guard
 *
 * Automatically begins a transaction on construction and rolls back on
 * destruction unless explicitly committed.
 */
class Transaction {
public:
    /**
     * @brief Begin a new transaction
     * @param conn Connection to use
     * @throws QueryError if transaction cannot be started
     */
    explicit Transaction(SqlConnection& conn);

    /**
     * @brief Rollback if not committed
     */
    ~Transaction();

    // Non-copyable, non-movable
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) = delete;
    Transaction& operator=(Transaction&&) = delete;

    /**
     * @brief Commit the transaction
     * @throws QueryError if commit fails
     *
     * After calling commit(), the transaction is completed and the destructor
     * will not rollback.
     */
    void commit();

    /**
     * @brief Rollback the transaction
     *
     * Explicitly rollback the transaction. This is optional as the destructor
     * will rollback if not committed.
     */
    void rollback();

private:
    SqlConnection& conn_;
    bool committed_;
    bool rolled_back_;
};

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

namespace util {

/**
 * @brief Escape a string for use in SQL
 * @param str String to escape
 * @return Escaped string
 */
std::string escapeSql(const std::string& str);

/**
 * @brief Quote an identifier (table or column name)
 * @param identifier Identifier to quote
 * @return Quoted identifier
 */
std::string quoteIdentifier(const std::string& identifier);

} // namespace util

} // namespace ado

#endif /* SQL_WRAPPER_H */
