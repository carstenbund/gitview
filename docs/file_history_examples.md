# File History Header Examples

This document shows what injected file history headers look like in different programming languages.

---

## Python Example

```python
# ==============================================================================
# FILE CHANGE HISTORY
# ==============================================================================
# Generated: 2026-01-18 15:30:00
# File: src/database/connection_pool.py
# Total changes: 47 commits
# Primary authors: Alice Chen (23), Bob Smith (15), Carol Davis (9)
#
# [Recent Changes - Last 10 of 47]
#
# 2026-01-15 10:23:45 | abc1234 | Alice Chen
#   Modified: process_request(), handle_error(), retry_with_backoff()
#   Added: exponential_backoff()
#   Changes: +45 lines, -12 lines
#
#   Summary: Implemented robust retry mechanism to handle transient network
#   failures. Added exponential backoff with max 5 retry attempts and
#   configurable timeout. This resolves production issues with connection
#   timeouts under high load.
#
# 2025-12-10 08:15:32 | xyz5678 | Bob Smith
#   Modified: init_db(), get_connection(), release_connection()
#   Added: ConnectionPool class
#   Changes: +23 lines, -8 lines
#
#   Summary: Refactored manual connection management to use connection pooling.
#   Improves performance by reusing connections instead of creating new ones
#   for each request. Reduces database connection overhead by ~60%.
#
# 2025-11-22 14:42:18 | def9012 | Alice Chen
#   Modified: execute_query(), handle_timeout()
#   Changes: +8 lines, -3 lines
#
#   Summary: Increased default query timeout from 30s to 60s based on
#   production metrics. Added better error messages for timeout scenarios.
#
# 2025-10-15 09:30:22 | ghi3456 | Carol Davis
#   Modified: validate_connection(), check_health()
#   Added: is_connection_alive()
#   Changes: +12 lines, -0 lines
#
#   Summary: Added health check mechanism to validate connections before use.
#   Prevents errors from stale connections in the pool.
#
# 2025-09-08 16:20:11 | jkl7890 | Bob Smith
#   Modified: close_all_connections()
#   Changes: +5 lines, -2 lines
#
#   Summary: Fixed resource leak where connections weren't properly closed on
#   shutdown. Added explicit cleanup in exception handlers.
#
# 2025-08-14 11:05:44 | mno1234 | Alice Chen
#   Modified: __init__(), configure()
#   Changes: +18 lines, -7 lines
#
#   Summary: Added configuration options for min/max pool size. Allows dynamic
#   scaling of connection pool based on load patterns.
#
# 2025-07-22 10:15:33 | pqr5678 | Carol Davis
#   Modified: acquire_connection()
#   Changes: +6 lines, -4 lines
#
#   Summary: Fixed race condition in connection acquisition under high
#   concurrency. Added lock to prevent multiple threads from grabbing same
#   connection.
#
# 2025-06-30 13:45:22 | stu9012 | Bob Smith
#   Modified: log_connection_stats()
#   Added: get_pool_metrics()
#   Changes: +22 lines, -0 lines
#
#   Summary: Added detailed metrics logging for connection pool monitoring.
#   Tracks active connections, queue length, and wait times.
#
# 2025-06-12 09:20:15 | vwx3456 | Alice Chen
#   Modified: handle_connection_error()
#   Changes: +10 lines, -5 lines
#
#   Summary: Improved error handling for connection failures. Distinguishes
#   between transient and permanent failures for better retry logic.
#
# 2025-05-28 14:30:08 | yza7890 | Carol Davis
#   Modified: create_connection()
#   Changes: +7 lines, -3 lines
#
#   Summary: Added SSL support for database connections. Enables encrypted
#   connections to production database.
#
# Full history: .gitview/file_histories/src/database/connection_pool.py.json
# ==============================================================================

"""Database connection pool manager with retry logic"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Manages a pool of database connections with health checking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = []
        self.max_size = config.get('max_size', 100)
        # ... rest of implementation
```

---

## JavaScript Example

```javascript
/*******************************************************************************
 * FILE CHANGE HISTORY
 *******************************************************************************
 * Generated: 2026-01-18 15:30:00
 * File: src/api/auth/jwt-manager.js
 * Total changes: 32 commits
 * Primary authors: David Lee (18), Emma Wilson (10), Frank Zhang (4)
 *
 * [Recent Changes - Last 5 of 32]
 *
 * 2026-01-12 11:45:23 | abc9876 | David Lee
 *   Modified: generateToken(), verifyToken()
 *   Added: refreshToken()
 *   Changes: +38 lines, -15 lines
 *
 *   Summary: Implemented JWT refresh token mechanism to extend session
 *   lifetime without requiring re-authentication. Tokens now expire after
 *   15 minutes but can be refreshed for up to 7 days.
 *
 * 2025-12-18 09:22:17 | def5432 | Emma Wilson
 *   Modified: verifyToken()
 *   Changes: +12 lines, -5 lines
 *
 *   Summary: Added token blacklist check to prevent use of revoked tokens.
 *   Integrates with Redis cache for fast blacklist lookups.
 *
 * 2025-11-30 14:10:05 | ghi1098 | David Lee
 *   Modified: generateToken()
 *   Changes: +8 lines, -3 lines
 *
 *   Summary: Switched from HS256 to RS256 algorithm for better security.
 *   Private key now stored in vault instead of environment variables.
 *
 * 2025-10-20 16:35:42 | jkl7654 | Frank Zhang
 *   Modified: extractTokenFromHeader()
 *   Changes: +6 lines, -2 lines
 *
 *   Summary: Fixed bug where Bearer token extraction failed with extra
 *   whitespace. Now properly handles malformed Authorization headers.
 *
 * 2025-09-15 10:05:18 | mno3210 | Emma Wilson
 *   Modified: generateToken()
 *   Added: includeCustomClaims()
 *   Changes: +18 lines, -0 lines
 *
 *   Summary: Added support for custom JWT claims to include user roles
 *   and permissions in token payload. Reduces database lookups on each
 *   request.
 *
 * Full history: .gitview/file_histories/src/api/auth/jwt-manager.js.json
 ******************************************************************************/

const jwt = require('jsonwebtoken');
const Redis = require('ioredis');
const { getVaultSecret } = require('../utils/vault');

/**
 * JWT token manager with refresh and revocation support
 */
class JWTManager {
  constructor(config) {
    this.config = config;
    this.redis = new Redis(config.redis);
    // ... rest of implementation
  }

  async generateToken(userId, claims = {}) {
    // ... implementation
  }
}

module.exports = JWTManager;
```

---

## Go Example

```go
////////////////////////////////////////////////////////////////////////////////
// FILE CHANGE HISTORY
////////////////////////////////////////////////////////////////////////////////
// Generated: 2026-01-18 15:30:00
// File: internal/cache/redis_client.go
// Total changes: 28 commits
// Primary authors: Grace Kim (15), Henry Park (9), Iris Chen (4)
//
// [Recent Changes - Last 5 of 28]
//
// 2026-01-10 13:25:40 | abc1111 | Grace Kim
//   Modified: Get(), Set(), Delete()
//   Added: GetWithExpiry(), SetNX()
//   Changes: +42 lines, -18 lines
//
//   Summary: Added atomic operations and expiry management. SetNX provides
//   set-if-not-exists semantics for distributed locking. GetWithExpiry returns
//   both value and remaining TTL for better cache management.
//
// 2025-12-05 10:15:22 | def2222 | Henry Park
//   Modified: Connect(), Close()
//   Changes: +15 lines, -8 lines
//
//   Summary: Implemented connection pooling with configurable pool size.
//   Significantly improves throughput under high concurrency by reusing
//   connections instead of creating new ones.
//
// 2025-11-18 16:40:33 | ghi3333 | Grace Kim
//   Modified: handleError()
//   Changes: +10 lines, -4 lines
//
//   Summary: Enhanced error handling to distinguish between connection errors
//   and data errors. Connection errors now trigger automatic reconnection.
//
// 2025-10-22 09:50:15 | jkl4444 | Iris Chen
//   Added: HealthCheck()
//   Changes: +18 lines, -0 lines
//
//   Summary: Added health check endpoint to verify Redis connectivity and
//   latency. Integrates with Kubernetes liveness probes.
//
// 2025-09-30 14:20:08 | mno5555 | Henry Park
//   Modified: Get(), Set()
//   Changes: +8 lines, -5 lines
//
//   Summary: Added context support for cancellation and timeout control.
//   Operations now respect context deadlines and can be cancelled gracefully.
//
// Full history: .gitview/file_histories/internal/cache/redis_client.go.json
////////////////////////////////////////////////////////////////////////////////

package cache

import (
    "context"
    "time"

    "github.com/go-redis/redis/v8"
)

// RedisClient wraps redis client with enhanced error handling
type RedisClient struct {
    client *redis.Client
    config *Config
}

// NewRedisClient creates a new Redis client with connection pooling
func NewRedisClient(config *Config) (*RedisClient, error) {
    // ... implementation
}
```

---

## Java Example

```java
/******************************************************************************
 * FILE CHANGE HISTORY
 ******************************************************************************
 * Generated: 2026-01-18 15:30:00
 * File: src/main/java/com/example/service/PaymentProcessor.java
 * Total changes: 56 commits
 * Primary authors: Jack Roberts (28), Kelly Moore (19), Lisa Thompson (9)
 *
 * [Recent Changes - Last 5 of 56]
 *
 * 2026-01-08 15:30:12 | abc7777 | Jack Roberts
 *   Modified: processPayment(), handleRefund()
 *   Added: validatePaymentMethod()
 *   Changes: +52 lines, -22 lines
 *
 *   Summary: Added comprehensive payment validation including card number
 *   verification, CVV validation, and fraud detection. Integrated with
 *   third-party fraud detection service for high-risk transactions.
 *
 * 2025-12-20 11:20:45 | def8888 | Kelly Moore
 *   Modified: recordTransaction()
 *   Changes: +15 lines, -8 lines
 *
 *   Summary: Enhanced transaction logging to include detailed audit trail.
 *   All payment attempts now logged with timestamps, amounts, and outcomes
 *   for compliance requirements.
 *
 * 2025-11-15 09:45:30 | ghi9999 | Jack Roberts
 *   Modified: handleRefund()
 *   Changes: +18 lines, -10 lines
 *
 *   Summary: Implemented partial refund support. Customers can now receive
 *   partial refunds without canceling entire order. Added refund reason
 *   tracking for analytics.
 *
 * 2025-10-28 14:10:22 | jkl0000 | Lisa Thompson
 *   Added: retryFailedPayment()
 *   Changes: +35 lines, -0 lines
 *
 *   Summary: Added automatic retry logic for transient payment failures.
 *   Retries with exponential backoff for network errors and temporary
 *   gateway issues. Maximum 3 retry attempts.
 *
 * 2025-09-12 16:55:18 | mno1111 | Kelly Moore
 *   Modified: processPayment()
 *   Changes: +12 lines, -5 lines
 *
 *   Summary: Fixed race condition in concurrent payment processing. Added
 *   transaction-level locking to prevent duplicate charges when users
 *   double-click submit button.
 *
 * Full history: .gitview/file_histories/src/main/java/com/example/service/PaymentProcessor.java.json
 *****************************************************************************/

package com.example.service;

import com.example.model.Payment;
import com.example.model.Refund;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Payment processing service with fraud detection and retry logic
 */
@Service
public class PaymentProcessor {

    private final PaymentGateway gateway;
    private final FraudDetectionService fraudDetection;

    public PaymentProcessor(PaymentGateway gateway, FraudDetectionService fraudDetection) {
        this.gateway = gateway;
        this.fraudDetection = fraudDetection;
    }

    @Transactional
    public PaymentResult processPayment(Payment payment) {
        // ... implementation
    }
}
```

---

## Compact Format Example (5 entries)

```python
# ==============================================================================
# FILE HISTORY - src/utils/logger.py
# ==============================================================================
# 2026-01-18 | abc123 | Alice | +12/-5 | Added structured logging with JSON output
# 2025-12-10 | def456 | Bob   | +8/-3  | Fixed log level configuration bug
# 2025-11-20 | ghi789 | Carol | +15/-0 | Added log rotation and compression
# 2025-10-15 | jkl012 | Alice | +6/-2  | Improved error message formatting
# 2025-09-08 | mno345 | Bob   | +10/-4 | Added request ID tracking for tracing
#
# Full history (23 changes): .gitview/file_histories/src/utils/logger.py.history
# ==============================================================================

import logging
import json
from datetime import datetime
# ... rest of code
```

---

## With Diff Snippets Example

```python
# ==============================================================================
# FILE CHANGE HISTORY
# ==============================================================================
# File: src/security/encryption.py
#
# [Recent Changes - Last 3 of 18]
#
# 2026-01-15 10:30:00 | abc1234 | Alice Chen
#   Modified: encrypt(), decrypt()
#   Added: rotate_keys()
#   Changes: +35 lines, -12 lines
#
#   Summary: Implemented automatic key rotation to enhance security. Keys now
#   rotate every 90 days with seamless decryption of old data using key
#   versioning.
#
#   Diff:
#   +    def rotate_keys(self):
#   +        """Rotate encryption keys while maintaining backward compatibility"""
#   +        old_key = self.current_key
#   +        new_key = self._generate_key()
#   +        self.key_history.append({'key': old_key, 'valid_until': datetime.now()})
#   +        self.current_key = new_key
#
#   -    def decrypt(self, data):
#   -        return self.cipher.decrypt(data)
#   +    def decrypt(self, data, key_version=None):
#   +        key = self._get_key_by_version(key_version) if key_version else self.current_key
#   +        return self.cipher.decrypt(data, key)
#
# 2025-12-01 14:20:15 | def5678 | Bob Smith
#   Modified: encrypt()
#   Changes: +8 lines, -3 lines
#
#   Summary: Switched from AES-128 to AES-256 for stronger encryption. All
#   new data encrypted with 256-bit keys for compliance with security policy.
#
#   Diff:
#   -    ALGORITHM = 'AES-128-CBC'
#   +    ALGORITHM = 'AES-256-CBC'
#   -    KEY_SIZE = 16
#   +    KEY_SIZE = 32
#
# Full history: .gitview/file_histories/src/security/encryption.py.json
# ==============================================================================

from cryptography.fernet import Fernet
import os

class Encryptor:
    # ... implementation
```

---

## JSON File Example

```json
{
  "file_path": "src/database/connection_pool.py",
  "current_path": "src/database/connection_pool.py",
  "first_commit": "2024-01-15T10:00:00Z",
  "first_commit_hash": "abc1234",
  "last_commit": "2026-01-18T15:30:00Z",
  "last_commit_hash": "xyz9876",
  "total_commits": 47,
  "total_lines_added": 523,
  "total_lines_removed": 178,
  "authors": [
    {"name": "Alice Chen", "email": "alice@example.com", "commits": 23},
    {"name": "Bob Smith", "email": "bob@example.com", "commits": 15},
    {"name": "Carol Davis", "email": "carol@example.com", "commits": 9}
  ],
  "previous_paths": [],
  "changes": [
    {
      "commit_hash": "abc1234",
      "commit_date": "2026-01-15T10:23:45Z",
      "author_name": "Alice Chen",
      "author_email": "alice@example.com",
      "commit_message": "Implement retry mechanism with exponential backoff",
      "lines_added": 45,
      "lines_removed": 12,
      "functions_added": ["exponential_backoff"],
      "functions_removed": [],
      "functions_modified": ["process_request", "handle_error", "retry_with_backoff"],
      "classes_added": [],
      "classes_modified": [],
      "diff_snippet": "...",
      "full_diff_available": true,
      "ai_summary": "Implemented robust retry mechanism to handle transient network failures. Added exponential backoff with max 5 retry attempts and configurable timeout. This resolves production issues with connection timeouts under high load.",
      "ai_summary_model": "gpt-4o-mini",
      "ai_summary_generated_at": "2026-01-18T15:30:00Z"
    }
  ]
}
```

---

## CLI Output Examples

### Generating Histories

```bash
$ gitview track-files /path/to/repo --patterns "*.py,*.js"

Analyzing git history...
╔════════════════════════════════════════════════════════════════╗
║               File History Tracking Progress                   ║
╚════════════════════════════════════════════════════════════════╝

Processing commits: [████████████████████████████] 1,247/1,247

Files tracked:
  Python:     142 files
  JavaScript:  89 files
  Total:      231 files

Changes extracted:
  Total commits with changes: 956
  Total file changes: 3,421
  Average changes per file: 14.8

Generated outputs:
  ✓ Companion .history files: 231
  ✓ JSON histories: 231
  ✓ Master index: 1

Checkpoint saved: commit xyz789 (2026-01-18 15:30:00)

Time elapsed: 42s
Cost: $0.00 (no AI summaries)

Next steps:
  • Add AI summaries: --with-ai
  • Inject headers: gitview inject-history .
  • Query file: gitview file-history <path>
```

### Injecting Headers

```bash
$ gitview inject-history src/ --max-entries 5

Injecting file histories...

╔════════════════════════════════════════════════════════════════╗
║                   Header Injection Progress                    ║
╚════════════════════════════════════════════════════════════════╝

Processing files: [████████████████████████████] 142/142

Results:
  ✓ Headers injected: 142 files
  ⊘ Skipped (no history): 0 files
  ⊘ Skipped (unsupported language): 0 files

File modifications:
  src/database/connection_pool.py    (+52 lines)
  src/api/auth/jwt_manager.py        (+48 lines)
  src/utils/logger.py                 (+35 lines)
  src/cache/redis_client.py           (+41 lines)
  ... (138 more files)

Total lines added: 5,847

⚠ Warning: Files have been modified with history headers
  To remove: gitview inject-history src/ --remove

Tip: Review changes before committing
  git diff src/
```

### Querying File History

```bash
$ gitview file-history src/database/connection_pool.py --recent 3

╔════════════════════════════════════════════════════════════════╗
║            File History: connection_pool.py                    ║
╚════════════════════════════════════════════════════════════════╝

Path: src/database/connection_pool.py
First seen: 2024-01-15 (commit abc1234)
Last modified: 2026-01-18 (commit xyz9876)

Statistics:
  Total commits: 47
  Lines added: 523
  Lines removed: 178
  Net change: +345 lines

Contributors:
  Alice Chen   23 commits (48.9%)
  Bob Smith    15 commits (31.9%)
  Carol Davis   9 commits (19.1%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Recent Changes - Last 3 of 47]

┌─────────────────────────────────────────────────────────────────┐
│ 2026-01-15 10:23:45 | abc1234 | Alice Chen                      │
├─────────────────────────────────────────────────────────────────┤
│ Implement retry mechanism with exponential backoff              │
│                                                                  │
│ Changes:                                                         │
│   • Modified: process_request(), handle_error()                 │
│   • Added: exponential_backoff()                                │
│   • +45 lines, -12 lines                                        │
│                                                                  │
│ Summary:                                                         │
│   Implemented robust retry mechanism to handle transient        │
│   network failures. Added exponential backoff with max 5 retry  │
│   attempts and configurable timeout. This resolves production   │
│   issues with connection timeouts under high load.              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2025-12-10 08:15:32 | xyz5678 | Bob Smith                       │
├─────────────────────────────────────────────────────────────────┤
│ Refactor to use connection pooling                              │
│                                                                  │
│ Changes:                                                         │
│   • Modified: init_db(), get_connection()                       │
│   • Added: ConnectionPool class                                 │
│   • +23 lines, -8 lines                                         │
│                                                                  │
│ Summary:                                                         │
│   Refactored manual connection management to use connection     │
│   pooling. Improves performance by reusing connections instead  │
│   of creating new ones for each request. Reduces database       │
│   connection overhead by ~60%.                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2025-11-22 14:42:18 | def9012 | Alice Chen                      │
├─────────────────────────────────────────────────────────────────┤
│ Increase default query timeout                                  │
│                                                                  │
│ Changes:                                                         │
│   • Modified: execute_query(), handle_timeout()                 │
│   • +8 lines, -3 lines                                          │
│                                                                  │
│ Summary:                                                         │
│   Increased default query timeout from 30s to 60s based on      │
│   production metrics. Added better error messages for timeout   │
│   scenarios.                                                     │
└─────────────────────────────────────────────────────────────────┘

Full history available:
  .gitview/file_histories/src/database/connection_pool.py.json

Export options:
  --format markdown > history.md
  --format json > history.json
```

---

## Integration with IDE

### VSCode Extension (Future)

```
file_history_vscode/
├── src/
│   ├── extension.ts          # Main extension
│   ├── historyProvider.ts    # TreeView provider
│   └── decorators.ts         # File decorations
└── package.json

Features:
  • TreeView showing file history in sidebar
  • Inline annotations showing last modifier
  • "Show History" command in context menu
  • Diff view for changes
  • "Inject History Header" command
```

### JetBrains Plugin (Future)

```
file_history_intellij/
├── src/
│   ├── FileHistoryToolWindow.kt
│   ├── HistoryInjectionAction.kt
│   └── HistoryAnnotator.kt
└── plugin.xml

Features:
  • Tool window with file history timeline
  • Gutter icons showing change count
  • Quick actions to inject/remove headers
  • Integration with VCS
```
