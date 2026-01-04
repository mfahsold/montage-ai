# Architectural Decisions Records (ADR)

## ADR-001: "Hardware Nah" Efficiency Strategy

**Date:** 2025-12-31
**Status:** Accepted

### Context
Montage AI is designed to run on a wide range of hardware, from powerful workstations to resource-constrained laptops and edge devices. We need to ensure that the application is as efficient as possible ("Hardware Nah" - close to hardware) to maximize performance and minimize resource waste.

### Decision
We will adopt a "Hardware Nah" philosophy across the entire stack, prioritizing efficiency, low overhead, and direct hardware access where possible.

### Implementation Details

#### 1. Web UI: Server-Sent Events (SSE) vs. Polling
*   **Decision:** Use SSE for real-time job status updates.
*   **Rationale:** Traditional polling (e.g., every 2 seconds) wastes CPU cycles and network bandwidth on both client and server, especially when no changes occur. SSE maintains a single open connection and pushes updates only when state changes.
*   **Impact:** Near-zero network overhead when idle. Lower CPU usage on client browsers.

#### 2. Process Priority Management
*   **Decision:** Run heavy rendering jobs with lower process priority (`os.nice(10)`).
*   **Rationale:** Video rendering is CPU-intensive and can make the system unresponsive. By lowering the priority (increasing niceness), we ensure that the Web UI and other interactive processes remain responsive even during heavy load.
*   **Impact:** Improved user experience during rendering.

#### 3. CLI: Lazy Loading
*   **Decision:** Use lazy imports for heavy libraries (e.g., `rich`) in the CLI.
*   **Rationale:** Python startup time can be slow if many libraries are imported at the top level. Most CLI commands (like `help` or `list`) don't need the full stack.
*   **Impact:** Instant CLI response time. Reduced memory footprint for simple commands.

#### 4. Cluster: Image Pull Policy
*   **Decision:** Use `imagePullPolicy: IfNotPresent` in Kubernetes jobs.
*   **Rationale:** `Always` pulling images wastes significant bandwidth and slows down job startup, especially for large AI/video images.
*   **Impact:** Faster job startup, reduced network traffic.

#### 5. DRY Environment Mapping
*   **Decision:** Centralize environment variable mapping in `src/montage_ai/env_mapper.py`.
*   **Rationale:** Both the CLI and Web UI need to pass options to the underlying `editor` module via environment variables. Duplicating this logic leads to bugs and inconsistencies.
*   **Impact:** Single source of truth for configuration mapping. Easier maintenance.

### Consequences
*   **Positive:** Lower resource usage, faster response times, better maintainability.
*   **Negative:** Slightly more complex implementation for SSE compared to simple polling.
