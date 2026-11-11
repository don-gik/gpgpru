use super::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum BatchKind {
    Compute(u64),
    Transfer,
    Timestamp,
}

#[derive(Debug)]
pub(crate) struct TaskBatch {
    pub(crate) kind: BatchKind,
    pub(crate) need_barrier: bool,
    pub(crate) src_access: vk::AccessFlags2,
    pub(crate) dst_access: vk::AccessFlags2,
    pub(crate) tasks: SmallVec<[Task; 16]>,
}

pub(crate) struct TaskQueue {
    queue: SegQueue<Task>,
    last_was_timestamp: Cell<bool>,
}

impl TaskQueue {
    pub(crate) fn new() -> Self {
        Self {
            queue: SegQueue::new(),
            last_was_timestamp: Cell::new(false),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> u16 {
        self.queue.len() as u16
    }

    pub(crate) fn push_task(&self, mut task: Task) {
        let len = self.queue.len();
        let is_timestamp = matches!(task.payload, TaskPayload::Timestamp { .. });

        if len > 0 {
            let prev_idx = (len - 1) as u32;

            // If previous task was timestamp or current task is timestamp
            // Then add DAG graph dep
            if is_timestamp || self.last_was_timestamp.get() {
                task.hazards.deps.push(prev_idx as u16);
            }
        }

        // Record current task
        self.last_was_timestamp.set(is_timestamp);
        self.queue.push(task);
    }

    pub(crate) fn drain(&self) -> Vec<Task> {
        let mut batched_tasks = Vec::with_capacity(self.queue.len());
        while let Some(task) = self.queue.pop() {
            batched_tasks.push(task);
        }
        batched_tasks
    }

    /// Sorts DAG topologically
    /// then merges compatible tasks
    pub(crate) fn drain_sort_and_coalesce(&self) -> Vec<TaskBatch> {
        let mut tasks = self.drain();
        let n = tasks.len();
        if n == 0 {
            return Vec::new();
        }

        if n == 1 {
            let task = tasks.pop().unwrap();
            let kind = task.kind();
            return vec![TaskBatch {
                kind,
                need_barrier: task.hazards.need_barrier,
                src_access: task.hazards.src_access,
                dst_access: task.hazards.dst_access,
                tasks: smallvec![task],
            }];
        }

        TASK_SCRATCH.with(|scratch_cell| {
            let scratch = &mut *scratch_cell.borrow_mut();
            scratch.prepare(n);

            // 1. Flattened Adjacency List (CSR Format) & In-Degree Calculation
            for task in &tasks {
                for &parent_idx in &task.hazards.deps {
                    scratch.adj_offsets[parent_idx as usize + 1] += 1;
                }
            }

            for i in 0..n {
                scratch.adj_offsets[i + 1] += scratch.adj_offsets[i];
            }

            scratch.adj_cursor.clear();
            scratch.adj_cursor.extend_from_slice(&scratch.adj_offsets);

            let total_edges = scratch.adj_offsets[n] as usize;
            scratch.adj_flat.clear();
            scratch.adj_flat.resize(total_edges, 0);

            for (i, task) in tasks.iter().enumerate() {
                scratch.in_degree[i] = task.hazards.deps.len() as u16;
                for &parent_idx in &task.hazards.deps {
                    let p = parent_idx as usize;
                    let cursor = scratch.adj_cursor[p] as usize;
                    scratch.adj_flat[cursor] = i as u16;
                    scratch.adj_cursor[p] += 1;
                }
            }

            // 2. Ready Queue
            let mut ready: SmallVec<[u16; 32]> = SmallVec::new();
            for (idx, &deg) in scratch.in_degree.iter().enumerate() {
                if deg == 0 {
                    ready.push(idx as u16);
                }
            }

            let mut current_pipeline: Option<u64> = None;

            // 3. Topology Sort with Pipeline Affinity Matching
            while !ready.is_empty() {
                let ready_pos = current_pipeline.and_then(|pipe| {
                    ready
                        .iter()
                        .position(|&idx| tasks[idx as usize].kind() == BatchKind::Compute(pipe))
                });

                let next_idx = match ready_pos {
                    Some(pos) => ready.swap_remove(pos),
                    None => {
                        let idx = ready.swap_remove(0);
                        match tasks[idx as usize].kind() {
                            BatchKind::Compute(pipe) => current_pipeline = Some(pipe),
                            BatchKind::Transfer => current_pipeline = None,
                            BatchKind::Timestamp => {}
                        }
                        idx
                    }
                } as usize;

                scratch.sorted_indices.push(next_idx);

                let start = scratch.adj_offsets[next_idx] as usize;
                let end = scratch.adj_offsets[next_idx + 1] as usize;
                for &child_idx in &scratch.adj_flat[start..end] {
                    let c = child_idx as usize;
                    scratch.in_degree[c] -= 1;
                    if scratch.in_degree[c] == 0 {
                        ready.push(child_idx);
                    }
                }
            }

            // Cyclic dependency fallback
            if scratch.sorted_indices.len() < n {
                scratch.visited.clear();
                scratch.visited.resize(n, false);
                for &idx in &scratch.sorted_indices {
                    scratch.visited[idx] = true;
                }
                for idx in 0..n {
                    if !scratch.visited[idx] {
                        scratch.sorted_indices.push(idx);
                    }
                }
            }

            // 4. Barrier Coalescing Pass (O(1) Direct Lookup)
            let mut batches: Vec<TaskBatch> = Vec::with_capacity(n);
            let mut task_options: Vec<Option<Task>> = tasks.into_iter().map(Some).collect();
            let mut current_batch_idx = 0usize;

            for &orig_idx in &scratch.sorted_indices {
                let task = task_options[orig_idx].take().unwrap();
                let task_kind = task.kind();

                let can_merge =
                    batches.last().map_or(false, |last_batch| {
                        last_batch.kind == task_kind
                            && !task.hazards.deps.iter().any(|&dep| {
                                scratch.task_to_batch[dep as usize] == current_batch_idx
                            })
                    });

                if can_merge {
                    let active_batch = batches.last_mut().unwrap();
                    active_batch.need_barrier |= task.hazards.need_barrier;
                    active_batch.src_access |= task.hazards.src_access;
                    active_batch.dst_access |= task.hazards.dst_access;
                    active_batch.tasks.push(task);
                    scratch.task_to_batch[orig_idx] = current_batch_idx;
                } else {
                    if !batches.is_empty() {
                        current_batch_idx += 1;
                    }
                    scratch.task_to_batch[orig_idx] = current_batch_idx;
                    batches.push(TaskBatch {
                        kind: task_kind,
                        need_barrier: task.hazards.need_barrier,
                        src_access: task.hazards.src_access,
                        dst_access: task.hazards.dst_access,
                        tasks: smallvec![task],
                    });
                }
            }

            batches
        })
    }
}
