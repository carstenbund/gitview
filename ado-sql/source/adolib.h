/**
 * @file adolib.h
 * @brief ADO Library - C Buffer-based Database Interface
 *
 * Provides a C API for database operations using buffer-based data structures.
 * This interface maintains backward compatibility with legacy applications while
 * providing modern SQL backend storage.
 */

#ifndef ADOLIB_H
#define ADOLIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define ADOLIB_VERSION_MAJOR 1
#define ADOLIB_VERSION_MINOR 0
#define ADOLIB_VERSION_PATCH 0

/* Buffer handle - opaque to users */
typedef struct ado_buffer* ADO_BUFFER;

/* Connection handle - opaque to users */
typedef struct ado_connection* ADO_CONNECTION;

/**
 * @brief Status codes returned by ADO functions
 */
typedef enum {
    ADO_OK = 0,               /**< Operation successful */
    ADO_ERROR = -1,           /**< General error */
    ADO_NOT_FOUND = -2,       /**< Record not found */
    ADO_INVALID_BUFFER = -3,  /**< Invalid buffer handle */
    ADO_INVALID_CONNECTION = -4, /**< Invalid connection handle */
    ADO_SQL_ERROR = -5,       /**< SQL execution error */
    ADO_MEMORY_ERROR = -6,    /**< Memory allocation error */
    ADO_SCHEMA_ERROR = -7,    /**< Schema mismatch error */
    ADO_TYPE_ERROR = -8       /**< Type conversion error */
} ado_status_t;

/* ============================================================================
 * Connection Management
 * ============================================================================ */

/**
 * @brief Connect to a database
 * @param database_path Path to the database file (SQLite) or connection string
 * @return Connection handle, or NULL on failure
 *
 * Opens a connection to the specified database. For SQLite, this is a file path.
 * The connection must be closed with ado_disconnect() when done.
 */
ADO_CONNECTION ado_connect(const char* database_path);

/**
 * @brief Disconnect from database
 * @param conn Connection handle to close
 *
 * Closes the database connection and frees associated resources.
 * Any uncommitted transactions will be rolled back.
 */
void ado_disconnect(ADO_CONNECTION conn);

/**
 * @brief Begin a new transaction
 * @param conn Connection handle
 * @return ADO_OK on success, error code otherwise
 *
 * Starts a new database transaction. All subsequent operations will be
 * part of this transaction until ado_commit() or ado_rollback() is called.
 */
ado_status_t ado_begin_transaction(ADO_CONNECTION conn);

/**
 * @brief Commit current transaction
 * @param conn Connection handle
 * @return ADO_OK on success, error code otherwise
 *
 * Commits all changes made in the current transaction.
 */
ado_status_t ado_commit(ADO_CONNECTION conn);

/**
 * @brief Rollback current transaction
 * @param conn Connection handle
 * @return ADO_OK on success, error code otherwise
 *
 * Rolls back all changes made in the current transaction.
 */
ado_status_t ado_rollback(ADO_CONNECTION conn);

/* ============================================================================
 * Buffer Operations
 * ============================================================================ */

/**
 * @brief Create a new buffer
 * @param size Size of the buffer in bytes
 * @return Buffer handle, or NULL on failure
 *
 * Allocates a new buffer of the specified size. The buffer must be freed
 * with ado_buffer_free() when no longer needed.
 */
ADO_BUFFER ado_buffer_create(size_t size);

/**
 * @brief Create a buffer for a specific table
 * @param table_name Name of the table
 * @return Buffer handle sized for the table schema, or NULL on failure
 *
 * Creates a buffer with the correct size for the specified table based on
 * the registered schema. Automatically sets the table association.
 */
ADO_BUFFER ado_buffer_create_for_table(const char* table_name);

/**
 * @brief Free a buffer
 * @param buf Buffer handle to free
 *
 * Frees the memory associated with the buffer.
 */
void ado_buffer_free(ADO_BUFFER buf);

/**
 * @brief Set a field value in a buffer
 * @param buf Buffer handle
 * @param field_name Name of the field
 * @param data Pointer to the data
 * @param data_len Length of the data in bytes
 * @return ADO_OK on success, error code otherwise
 *
 * Sets the value of a named field in the buffer. The data is copied into
 * the buffer at the appropriate offset based on the schema.
 */
ado_status_t ado_buffer_set_field(ADO_BUFFER buf, const char* field_name,
                                   const void* data, size_t data_len);

/**
 * @brief Get a field value from a buffer
 * @param buf Buffer handle
 * @param field_name Name of the field
 * @param data Pointer to receive the data
 * @param data_len Pointer to size; updated with actual length
 * @return ADO_OK on success, error code otherwise
 *
 * Retrieves the value of a named field from the buffer. The caller must
 * provide a buffer large enough to hold the data.
 */
ado_status_t ado_buffer_get_field(ADO_BUFFER buf, const char* field_name,
                                   void* data, size_t* data_len);

/**
 * @brief Get buffer size
 * @param buf Buffer handle
 * @return Size of buffer in bytes, or 0 on error
 */
size_t ado_buffer_size(ADO_BUFFER buf);

/**
 * @brief Associate buffer with a table
 * @param buf Buffer handle
 * @param table_name Name of the table
 * @return ADO_OK on success, error code otherwise
 */
ado_status_t ado_buffer_set_table(ADO_BUFFER buf, const char* table_name);

/* ============================================================================
 * CRUD Operations
 * ============================================================================ */

/**
 * @brief Insert a record into a table
 * @param conn Connection handle
 * @param table_name Name of the table
 * @param buf Buffer containing the data to insert
 * @return ADO_OK on success, error code otherwise
 *
 * Inserts a new record into the specified table using the data from the buffer.
 */
ado_status_t ado_insert(ADO_CONNECTION conn, const char* table_name,
                        ADO_BUFFER buf);

/**
 * @brief Update records in a table
 * @param conn Connection handle
 * @param table_name Name of the table
 * @param where_clause SQL WHERE clause (without "WHERE" keyword), or NULL for all
 * @param buf Buffer containing the updated data
 * @return ADO_OK on success, error code otherwise
 *
 * Updates existing records matching the WHERE clause with data from the buffer.
 */
ado_status_t ado_update(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause, ADO_BUFFER buf);

/**
 * @brief Delete records from a table
 * @param conn Connection handle
 * @param table_name Name of the table
 * @param where_clause SQL WHERE clause (without "WHERE" keyword), or NULL for all
 * @return ADO_OK on success, error code otherwise
 *
 * Deletes records matching the WHERE clause from the table.
 */
ado_status_t ado_delete(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause);

/**
 * @brief Select records from a table
 * @param conn Connection handle
 * @param table_name Name of the table
 * @param where_clause SQL WHERE clause (without "WHERE" keyword), or NULL for all
 * @param results Pointer to receive array of result buffers
 * @param num_results Pointer to receive the number of results
 * @return ADO_OK on success, error code otherwise
 *
 * Retrieves records matching the WHERE clause. The caller must free each
 * buffer in the results array with ado_buffer_free(), and free the array itself.
 */
ado_status_t ado_select(ADO_CONNECTION conn, const char* table_name,
                        const char* where_clause, ADO_BUFFER** results,
                        size_t* num_results);

/**
 * @brief Free an array of result buffers
 * @param results Array of buffer handles
 * @param num_results Number of buffers in the array
 *
 * Convenience function to free all buffers in a result set.
 */
void ado_free_results(ADO_BUFFER* results, size_t num_results);

/**
 * @brief Get the last inserted row ID
 * @param conn Connection handle
 * @return Last insert ID, or -1 on error
 */
int64_t ado_last_insert_id(ADO_CONNECTION conn);

/* ============================================================================
 * Schema Management
 * ============================================================================ */

/**
 * @brief Load schema from a .def file
 * @param def_file_path Path to the .def file
 * @return ADO_OK on success, error code otherwise
 *
 * Parses a schema definition file and registers the table schema.
 */
ado_status_t ado_load_schema(const char* def_file_path);

/**
 * @brief Create a table from registered schema
 * @param conn Connection handle
 * @param table_name Name of the table to create
 * @return ADO_OK on success, error code otherwise
 *
 * Creates a table in the database based on a previously registered schema.
 */
ado_status_t ado_create_table(ADO_CONNECTION conn, const char* table_name);

/**
 * @brief Check if a table exists
 * @param conn Connection handle
 * @param table_name Name of the table
 * @return 1 if exists, 0 if not, -1 on error
 */
int ado_table_exists(ADO_CONNECTION conn, const char* table_name);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * @brief Get the last error message
 * @param conn Connection handle
 * @return Error message string, or NULL if no error
 *
 * Returns a human-readable description of the last error that occurred.
 * The string is valid until the next ADO operation on this connection.
 */
const char* ado_get_error(ADO_CONNECTION conn);

/**
 * @brief Clear the last error
 * @param conn Connection handle
 *
 * Clears the error state for the connection.
 */
void ado_clear_error(ADO_CONNECTION conn);

/**
 * @brief Get error message for status code
 * @param status Status code
 * @return Error message string
 */
const char* ado_status_string(ado_status_t status);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get library version
 * @param major Pointer to receive major version
 * @param minor Pointer to receive minor version
 * @param patch Pointer to receive patch version
 */
void ado_version(int* major, int* minor, int* patch);

/**
 * @brief Initialize the ADO library
 * @return ADO_OK on success, error code otherwise
 *
 * Initializes global library state. Must be called before any other ADO functions.
 */
ado_status_t ado_initialize(void);

/**
 * @brief Shutdown the ADO library
 *
 * Cleans up global library state. Should be called when done with the library.
 */
void ado_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* ADOLIB_H */
