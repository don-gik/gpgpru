// use super::*;

use std::cell::RefCell;

// Single allocation per thread. (Dispatch is done by single thread :)
thread_local! {
    pub(crate) static TASK_SCRATCH: RefCell<TaskScratchSpace> = RefCell::new(TaskScratchSpace::default());
}

#[derive(Default)]
pub(crate) struct TaskScratchSpace {
    pub in_degree: Vec<u16>,
    pub adj_offsets: Vec<u16>,
    pub adj_cursor: Vec<u16>,
    pub adj_flat: Vec<u16>,
    pub sorted_indices: Vec<usize>,
    pub task_to_batch: Vec<usize>,
    pub visited: Vec<bool>,
}

impl TaskScratchSpace {
    pub fn prepare(&mut self, n: usize) {
        self.in_degree.clear();
        self.in_degree.resize(n, 0);

        self.adj_offsets.clear();
        self.adj_offsets.resize(n + 1, 0);

        self.sorted_indices.clear();
        self.sorted_indices.reserve(n);

        self.task_to_batch.clear();
        self.task_to_batch.resize(n, usize::MAX);
    }
}
