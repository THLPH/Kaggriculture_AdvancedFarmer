# Column-Based Worker Allocation Strategy

## Overview

This strategy proposes a deterministic, spatially partitioned approach to worker allocation in the Kaggriculture environment. Each 5×5 farming region is divided into five columns, with individual workers assigned responsibility for a designated column.

The objective is to establish consistent work boundaries, minimize overlapping task assignments, and coordinate farming operations through predefined spatial responsibilities.

## Worker Assignment

Workers are assigned columns according to their hiring order.

- The primary farmer is assigned the leftmost column of the starting 5×5 region.
- Each subsequently hired worker is assigned a designated column, progressing from left to right.
- Workers are directed toward their assigned columns before beginning farming operations.
- Once every column has an assigned worker, no additional workers are required for that region under this allocation policy.

## Positioning and Navigation

If a worker's initial position is occupied or does not correspond to its designated working area, the worker is repositioned horizontally along the east–west axis until it reaches its assigned column.

After reaching the designated column, the worker performs available farming tasks within its assigned area.

Following task completion, the worker may navigate vertically along the north–south axis to access tiles within the same column that have not yet been processed.

## Farm Expansion

When additional farming regions become available, the same column-based allocation procedure is applied to the expanded area.

Worker responsibilities are established using the same assignment and navigation principles.

## Design Considerations

The proposed approach emphasizes:

- **Spatial partitioning:** Dividing farming regions into clearly defined worker responsibilities.
- **Deterministic allocation:** Assigning workers according to a predefined ordering.
- **Task isolation:** Reducing overlapping assignments across different workers.
- **Structured navigation:** Restricting routine work traversal primarily to assigned columns.
- **Scalability:** Reapplying the allocation scheme as additional farming regions become available.

## Implementation Status

This document describes a proposed worker-allocation strategy. Its computational behavior, task-completion efficiency, and performance relative to alternative scheduling approaches require implementation and experimental validation.
