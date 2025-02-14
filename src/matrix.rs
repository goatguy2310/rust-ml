use std::{
    cell::RefCell, collections::HashSet, rc::Rc,
    hash::{Hash, Hasher},
    fmt::{self, Display, Formatter},
};

#[derive(Debug, Clone)]
pub struct Matrix(pub Rc<RefCell<RawMatrix>>);

#[derive(Debug, Clone)]
pub struct RawMatrix {
    rows: usize,
    cols: usize,

    data: Vec<f64>,
    grad: Vec<f64>,
    op: String,
    label: String,
    children: Vec<Matrix>,
}

// implement hash, eq, and display for Value
impl Hash for Matrix {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = Rc::as_ptr(&self.0);
        (ptr as usize).hash(state);
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Matrix {}