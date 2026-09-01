// src/device/command/queue.rs

use super::Task;

use crossbeam_queue::SegQueue;

#[allow(dead_code)]
pub(crate) struct TaskQueue {
    queue: SegQueue<Task>,
}

impl TaskQueue {
    pub(crate) fn new() -> Self {
        Self {
            queue: SegQueue::new(),
        }
    }

    pub(crate) fn push_task(&self, task: Task) {
        let _ = self.queue.push(task);
    }

    pub(crate) fn drain_and_sort(&self) -> Vec<Task> {
        let mut batched_tasks = Vec::with_capacity(self.queue.len());

        while let Some(task) = self.queue.pop() {
            batched_tasks.push(task);
        }

        batched_tasks.sort_by_key(|t| t.pipeline_id);

        batched_tasks
    }
}
