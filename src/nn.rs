use crate::value::*;

use rand::prelude::*;

// a single neuron
pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let w = (0..nin).map(|_| Value::new(thread_rng().gen_range(-1.0..1.0))).collect();
        let b = Value::new(thread_rng().gen_range(-1.0..1.0));
        Neuron {
            w,
            b
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let mut y = Value::new(self.b.get_data());
        for i in 0..self.w.len() {
            y = Value::add(&y, &Value::mul(&self.w[i], &x[i]));
        }
        y = Value::tanh(&y);
        return y;
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

// a layer of neurons
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer {
            neurons
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(&x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

// multiple layers of neurons
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(sz: &Vec<usize>) -> Self {
        let layers = sz.windows(2).map(|n| Layer::new(n[0], n[1])).collect();
        MLP {
            layers
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let mut y = x.clone();
        for l in &self.layers {
            y = l.forward(&y);
        }
        y
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(0.0);
        }
    }
}