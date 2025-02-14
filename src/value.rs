use std::{
    cell::RefCell, collections::HashSet, rc::Rc,
    hash::{Hash, Hasher},
    fmt::{self, Display, Formatter},
};

// Value struct for automatic differentiation
// using Rc and RefCell for sharing multiple pointers and mutable references
#[derive(Debug, Clone)]
pub struct Value(pub Rc<RefCell<RawValue>>);

// RawValue struct for the actual data
#[derive(Debug, Clone)]
pub struct RawValue {
    pub data: f64,
    pub grad: f64,
    pub op: String,
    pub label: String,
    pub children: Vec<Value>,

    pub extra: f64,
}

// implement hash, eq, and display for Value
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = Rc::as_ptr(&self.0);
        (ptr as usize).hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Value({} {})", self.get_data(), self.get_grad())
    }
}

impl Value {
    // constructor for Value
    pub fn new(data: f64) -> Value {
        return Value(Rc::new(RefCell::new(RawValue {
            data,
            grad: 0.0,
            op: "".to_string(),
            label: "".to_string(),
            children: vec![],
            extra: 0.0
        })));
    }

    // constructor for Value when made from an operator
    pub fn new_for_op(data: f64, op: &str, children: Vec<Value>, extra: f64) -> Value {
        return Value(Rc::new(RefCell::new(RawValue {
            data,
            grad: 0.0,
            op: op.to_string(),
            label: "".to_string(),
            children: children,
            extra
        })));
    }

    // getters and setters, and update
    pub fn get_data(&self) -> f64 {
        return self.0.borrow().data;
    }

    pub fn update_data(&self, data: f64) {
        self.0.borrow_mut().data += data;
    }

    pub fn get_grad(&self) -> f64 {
        return self.0.borrow().grad;
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn update_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad;
    }

    pub fn get_children(&self) -> Vec<Value> {
        return self.0.borrow().children.clone();
    }

    // get an rc pointer to the value, not cloning the value to another one
    pub fn clone_rc(&self) -> Value {
        return Value(Rc::clone(&self.0));
    }

    pub fn add(v1: &Value, v2: &Value) -> Value {
        return Value::new_for_op(
            v1.get_data() + v2.get_data(),
            "+",
            vec![v1.clone_rc(), v2.clone_rc()],
            0.0
        );
    }

    // a - b = a + (-b)
    pub fn sub(v1: &Value, v2: &Value) -> Value {
        return Self::add(&v1, &Self::neg(&v2));
    }

    pub fn mul(v1: &Value, v2: &Value) -> Value {
        return Value::new_for_op(
            v1.get_data() * v2.get_data(),
            "*",
            vec![v1.clone_rc(), v2.clone_rc()],
            0.0
        );
    }

    // a / b = a * (1 / b) = a * b^-1
    pub fn div(v1: &Value, v2: &Value) -> Value {
        return Self::mul(&v1, &Self::pow(&v2, -1.0));
    }

    pub fn neg(v1: &Value) -> Value {
        return Value::mul(&v1, &Value::new(-1.0));
    }

    pub fn pow(v1: &Value, p: f64) -> Value {
        return Value::new_for_op(
            v1.get_data().powf(p),
            format!("pow({})", p).as_str(),
            vec![v1.clone_rc()],
            p
        );
    }

    pub fn exp(val: &Value) -> Value {
        return Value::new_for_op(
            val.get_data().exp(),
            "exp",
            vec![val.clone_rc()],
            0.0
        );
    }

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    pub fn tanh(val: &Value) -> Value {
        let e = Value::exp(&Value::mul(&val, &Value::new(2.0)));
        return Value::div(&Value::sub(&e, &Value::new(1.0)), &Value::add(&e, &Value::new(1.0)));
    }

    // backward pass for the current node
    pub fn _backward(&self) {
        let val = self.0.borrow();
        let operation = val.op.as_str();
        match operation {
            "+" => {
                val.children[0].update_grad(val.grad);
                val.children[1].update_grad(val.grad);
            },
            "*" => {
                val.children[0].update_grad(val.grad * val.children[1].get_data());
                val.children[1].update_grad(val.grad * val.children[0].get_data());
            },
            "exp" => {
                val.children[0].update_grad(val.grad * val.data);
            },
            // match with anything starts with pow(
            s if s.starts_with("pow(") => {
                let p: f64 = val.extra;
                val.children[0].update_grad(val.grad * p * val.children[0].get_data().powf(p - 1.0));
            },

            _ => {},
        }
    }

    // backward pass for the entire graph
    pub fn backward(&self) {
        // find the topo sort
        let mut topo_sort: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        // iterative dfs
        let mut stack: Vec<Value> = vec![self.clone_rc()];
        while !stack.is_empty() {
            let node = stack[stack.len() - 1].clone_rc();
            if !visited.contains(&node) {
                visited.insert(node.clone());
                for child in node.get_children() {
                    if !visited.contains(&child) {
                        stack.push(child);
                    }
                }
            } else {
                topo_sort.push(node);
                stack.pop();
            }
        }

        self.set_grad(1.0);
        // backward pass
        for node in topo_sort.iter().rev() {
            // println!("{} {} {}", node.0.borrow().label, node.get_data(), node.get_grad());
            node._backward();
        }
    }
}
