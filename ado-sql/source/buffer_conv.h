/**
 * @file buffer_conv.h
 * @brief Buffer/SQL Conversion Layer
 *
 * Provides conversion between C buffer structures and SQL rows, schema
 * management, and .def file parsing.
 */

#ifndef BUFFER_CONV_H
#define BUFFER_CONV_H

#include "adolib.h"
#include "sql_wrapper.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace ado {
namespace conversion {

/* ============================================================================
 * Buffer Schema Definition
 * ============================================================================ */

/**
 * @brief Describes the structure of a buffer/table
 */
struct BufferSchema {
    /**
     * @brief Field definition in a buffer
     */
    struct Field {
        enum Type {
            INT8,      /**< 8-bit signed integer */
            INT16,     /**< 16-bit signed integer */
            INT32,     /**< 32-bit signed integer */
            INT64,     /**< 64-bit signed integer */
            UINT8,     /**< 8-bit unsigned integer */
            UINT16,    /**< 16-bit unsigned integer */
            UINT32,    /**< 32-bit unsigned integer */
            UINT64,    /**< 64-bit unsigned integer */
            FLOAT,     /**< 32-bit floating point */
            DOUBLE,    /**< 64-bit floating point */
            STRING,    /**< Fixed-size string (null-terminated) */
            BLOB,      /**< Binary data */
            BOOL       /**< Boolean (stored as INT8) */
        };

        std::string name;        /**< Field name */
        Type type;               /**< Field data type */
        size_t offset;           /**< Offset in buffer (bytes) */
        size_t size;             /**< Size of field (bytes) */
        bool primary_key;        /**< Is this the primary key? */
        bool nullable;           /**< Can this field be NULL? */
        std::string default_val; /**< Default value (if any) */

        Field()
            : type(INT32)
            , offset(0)
            , size(0)
            , primary_key(false)
            , nullable(true)
        {}
    };

    std::string table_name;      /**< Name of the table */
    std::vector<Field> fields;   /**< List of fields in order */
    size_t total_size;           /**< Total buffer size (bytes) */

    BufferSchema()
        : total_size(0)
    {}

    /**
     * @brief Find a field by name
     * @param name Field name to find
     * @return Pointer to field, or nullptr if not found
     */
    const Field* findField(const std::string& name) const;

    /**
     * @brief Get field by index
     * @param index Field index
     * @return Reference to field
     * @throws std::out_of_range if index invalid
     */
    const Field& getField(size_t index) const;

    /**
     * @brief Get number of fields
     */
    size_t fieldCount() const { return fields.size(); }

    /**
     * @brief Get primary key field
     * @return Pointer to primary key field, or nullptr if none
     */
    const Field* getPrimaryKey() const;
};

/* ============================================================================
 * Schema Registry
 * ============================================================================ */

/**
 * @brief Global registry of table schemas
 *
 * Singleton that maintains mappings from table names to buffer schemas.
 * Thread-safe for read operations after initialization.
 */
class SchemaRegistry {
public:
    /**
     * @brief Get the singleton instance
     */
    static SchemaRegistry& instance();

    /**
     * @brief Register a schema for a table
     * @param table_name Name of the table
     * @param schema Buffer schema definition
     */
    void registerSchema(const std::string& table_name,
                       const BufferSchema& schema);

    /**
     * @brief Get schema for a table
     * @param table_name Name of the table
     * @return Buffer schema
     * @throws SchemaError if table not found
     */
    const BufferSchema& getSchema(const std::string& table_name) const;

    /**
     * @brief Check if schema exists for a table
     * @param table_name Name of the table
     * @return true if schema exists
     */
    bool hasSchema(const std::string& table_name) const;

    /**
     * @brief Get list of all registered table names
     */
    std::vector<std::string> listTables() const;

    /**
     * @brief Clear all registered schemas (for testing)
     */
    void clear();

private:
    SchemaRegistry() = default;
    ~SchemaRegistry() = default;

    // Non-copyable, non-movable
    SchemaRegistry(const SchemaRegistry&) = delete;
    SchemaRegistry& operator=(const SchemaRegistry&) = delete;

    std::map<std::string, BufferSchema> schemas_;
};

/* ============================================================================
 * Buffer <-> SQL Conversion Functions
 * ============================================================================ */

/**
 * @brief Convert buffer to SQL row
 * @param buf Buffer handle
 * @param table_name Name of the table (for schema lookup)
 * @return SqlRow containing the buffer data
 * @throws SchemaError if table schema not found
 * @throws TypeError if conversion fails
 */
SqlRow bufferToRow(ADO_BUFFER buf, const std::string& table_name);

/**
 * @brief Convert SQL row to buffer
 * @param row SQL row data
 * @param table_name Name of the table (for schema lookup)
 * @return Buffer handle containing the row data
 * @throws SchemaError if table schema not found
 * @throws TypeError if conversion fails
 */
ADO_BUFFER rowToBuffer(const SqlRow& row, const std::string& table_name);

/**
 * @brief Get field value from buffer
 * @param buf Buffer handle
 * @param field_name Name of the field
 * @param field_info Field schema information
 * @return SqlValue containing the field data
 * @throws TypeError if conversion fails
 */
SqlValue bufferGetField(ADO_BUFFER buf,
                       const std::string& field_name,
                       const BufferSchema::Field& field_info);

/**
 * @brief Set field value in buffer
 * @param buf Buffer handle
 * @param field_name Name of the field
 * @param value Value to set
 * @param field_info Field schema information
 * @throws TypeError if conversion fails
 */
void bufferSetField(ADO_BUFFER buf,
                   const std::string& field_name,
                   const SqlValue& value,
                   const BufferSchema::Field& field_info);

/* ============================================================================
 * Schema File Parsing (.def files)
 * ============================================================================ */

/**
 * @brief Parse a .def file and return the schema
 * @param def_file_path Path to the .def file
 * @return BufferSchema parsed from the file
 * @throws SchemaError if parsing fails
 *
 * Example .def file format:
 * ```
 * # Comment
 * TABLE users
 * {
 *     FIELD id         INT64    PRIMARY_KEY
 *     FIELD name       STRING   SIZE=100
 *     FIELD email      STRING   SIZE=255 NULLABLE
 *     FIELD balance    DOUBLE   DEFAULT=0.0
 *     FIELD created_at INT64
 *     FIELD is_active  BOOL     DEFAULT=true
 * }
 * ```
 */
BufferSchema parseDefFile(const std::string& def_file_path);

/**
 * @brief Parse .def file content from string
 * @param def_content Content of .def file
 * @return BufferSchema parsed from the content
 * @throws SchemaError if parsing fails
 */
BufferSchema parseDefString(const std::string& def_content);

/**
 * @brief Load and register schema from .def file
 * @param def_file_path Path to the .def file
 * @return Table name from the schema
 * @throws SchemaError if parsing or registration fails
 */
std::string loadSchemaFromDef(const std::string& def_file_path);

/**
 * @brief Load all .def files from a directory
 * @param directory Path to directory containing .def files
 * @return Number of schemas loaded
 * @throws SchemaError if any file fails to parse
 */
int loadAllSchemas(const std::string& directory);

/* ============================================================================
 * SQL DDL Generation
 * ============================================================================ */

/**
 * @brief Generate CREATE TABLE SQL from schema
 * @param schema Buffer schema
 * @param if_not_exists Add IF NOT EXISTS clause
 * @return SQL CREATE TABLE statement
 */
std::string generateCreateTableSQL(const BufferSchema& schema,
                                  bool if_not_exists = true);

/**
 * @brief Generate INSERT SQL from schema
 * @param schema Buffer schema
 * @return SQL INSERT statement with placeholders
 */
std::string generateInsertSQL(const BufferSchema& schema);

/**
 * @brief Generate UPDATE SQL from schema
 * @param schema Buffer schema
 * @param where_clause WHERE clause (without "WHERE" keyword)
 * @return SQL UPDATE statement with placeholders
 */
std::string generateUpdateSQL(const BufferSchema& schema,
                             const std::string& where_clause);

/**
 * @brief Generate SELECT SQL from schema
 * @param schema Buffer schema
 * @param where_clause WHERE clause (without "WHERE" keyword), or empty
 * @return SQL SELECT statement
 */
std::string generateSelectSQL(const BufferSchema& schema,
                             const std::string& where_clause);

/**
 * @brief Generate DELETE SQL from schema
 * @param table_name Table name
 * @param where_clause WHERE clause (without "WHERE" keyword)
 * @return SQL DELETE statement
 */
std::string generateDeleteSQL(const std::string& table_name,
                             const std::string& where_clause);

/* ============================================================================
 * Type Conversion Utilities
 * ============================================================================ */

namespace util {

/**
 * @brief Get SQL type name for a buffer field type
 * @param type Field type
 * @return SQL type name (e.g., "INTEGER", "TEXT", "REAL", "BLOB")
 */
std::string getSqlTypeName(BufferSchema::Field::Type type);

/**
 * @brief Get buffer field type from string
 * @param type_str Type string (e.g., "INT32", "STRING", "DOUBLE")
 * @return Field type enum
 * @throws SchemaError if type string is invalid
 */
BufferSchema::Field::Type parseFieldType(const std::string& type_str);

/**
 * @brief Get size for a field type
 * @param type Field type
 * @return Size in bytes (0 for variable-size types like STRING)
 */
size_t getTypeSize(BufferSchema::Field::Type type);

/**
 * @brief Calculate total buffer size with alignment
 * @param schema Buffer schema
 * @return Total size in bytes with proper alignment
 */
size_t calculateBufferSize(const BufferSchema& schema);

/**
 * @brief Align offset to proper boundary
 * @param offset Current offset
 * @param alignment Alignment requirement (power of 2)
 * @return Aligned offset
 */
size_t alignOffset(size_t offset, size_t alignment);

} // namespace util

} // namespace conversion
} // namespace ado

#endif /* BUFFER_CONV_H */
