// src/device/command/queue.rs

use super::Task;
use ash::vk::AccessFlags2;
use crossbeam_queue::SegQueue;
use smallvec::{SmallVec, smallvec};

#[derive(Debug)]
pub(crate) struct TaskBatch {
    pub(crate) pipeline_id: u64,
    pub(crate) need_barrier: bool,
    pub(crate) src_access: AccessFlags2,
    pub(crate) dst_access: AccessFlags2,
    pub(crate) tasks: SmallVec<[Task; 8]>,
}

pub(crate) struct TaskQueue {
    queue: SegQueue<Task>,
}

impl TaskQueue {
    pub(crate) fn new() -> Self {
        Self {
            queue: SegQueue::new(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> u16 {
        self.queue.len() as u16
    }

    pub(crate) fn push_task(&self, task: Task) {
        let _ = self.queue.push(task);
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
            return vec![TaskBatch {
                pipeline_id: task.pipeline_id,
                need_barrier: task.hazards.need_barrier,
                src_access: task.hazards.src_access,
                dst_access: task.hazards.dst_access,
                tasks: smallvec![task],
            }];
        }

        // 1. Flattened Adjacency List (CSR Format) & In-Degree Calculation
        let mut in_degree = vec![0u16; n];
        let mut adj_offsets = vec![0u16; n + 1];

        for task in &tasks {
            for &parent_idx in &task.hazards.deps {
                adj_offsets[parent_idx as usize + 1] += 1;
            }
        }

        for i in 0..n {
            adj_offsets[i + 1] += adj_offsets[i];
        }

        let mut adj_cursor = adj_offsets.clone();
        let mut adj_flat = vec![0u16; adj_offsets[n] as usize];

        for (i, task) in tasks.iter().enumerate() {
            in_degree[i] = task.hazards.deps.len() as u16;
            for &parent_idx in &task.hazards.deps {
                let p = parent_idx as usize;
                adj_flat[adj_cursor[p] as usize] = i as u16;
                adj_cursor[p] += 1;
            }
        }

        // 2. Stack-allocated Ready Queue
        let mut ready: SmallVec<[u16; 32]> = SmallVec::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(idx as u16);
            }
        }

        let mut sorted_indices = Vec::with_capacity(n);
        let mut current_pipeline: Option<u64> = None;

        // 3. Topology Sort with Pipeline Affinity Matching
        while !ready.is_empty() {
            let ready_pos = current_pipeline.and_then(|pipe| {
                ready
                    .iter()
                    .position(|&idx| tasks[idx as usize].pipeline_id == pipe)
            });

            let next_idx = match ready_pos {
                Some(pos) => ready.swap_remove(pos),
                None => {
                    let idx = ready.swap_remove(0);
                    current_pipeline = Some(tasks[idx as usize].pipeline_id);
                    idx
                }
            } as usize;

            sorted_indices.push(next_idx);

            let start = adj_offsets[next_idx] as usize;
            let end = adj_offsets[next_idx + 1] as usize;
            for &child_idx in &adj_flat[start..end] {
                let c = child_idx as usize;
                in_degree[c] -= 1;
                if in_degree[c] == 0 {
                    ready.push(child_idx);
                }
            }
        }

        // Cyclic dependency fallback
        if sorted_indices.len() < n {
            let mut visited = vec![false; n];
            for &idx in &sorted_indices {
                visited[idx] = true;
            }
            for idx in 0..n {
                if !visited[idx] {
                    sorted_indices.push(idx);
                }
            }
        }

        // 4. Barrier Coalescing Pass
        let mut batches: Vec<TaskBatch> = Vec::with_capacity(n);
        let mut current_batch_indices: SmallVec<[u16; 8]> = SmallVec::new();

        // Wrap tasks in Option for fast ownership transfer without clone/permute
        let mut task_options: Vec<Option<Task>> = tasks.into_iter().map(Some).collect();

        for &orig_idx in &sorted_indices {
            let task = task_options[orig_idx].take().unwrap();

            let can_merge = batches.last().map_or(false, |last_batch| {
                last_batch.pipeline_id == task.pipeline_id
                    && !task
                        .hazards
                        .deps
                        .iter()
                        .any(|dep| current_batch_indices.contains(dep))
            });

            if can_merge {
                let active_batch = batches.last_mut().unwrap();
                active_batch.need_barrier |= task.hazards.need_barrier;
                active_batch.src_access |= task.hazards.src_access;
                active_batch.dst_access |= task.hazards.dst_access;
                active_batch.tasks.push(task);
                current_batch_indices.push(orig_idx as u16);
            } else {
                current_batch_indices.clear();
                current_batch_indices.push(orig_idx as u16);
                batches.push(TaskBatch {
                    pipeline_id: task.pipeline_id,
                    need_barrier: task.hazards.need_barrier,
                    src_access: task.hazards.src_access,
                    dst_access: task.hazards.dst_access,
                    tasks: smallvec![task],
                });
            }
        }

        batches
    }
}
