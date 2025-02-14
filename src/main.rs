mod value;
mod matrix;
mod nn;

use value::Value;
use nn::MLP;

fn main() {
    // testing the value library
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = Value::new(3.0);

    let d = Value::mul(&a, &Value::add(&b, &c));
    let e = Value::mul(&d, &a);

    a.0.borrow_mut().label = "a".to_string();
    b.0.borrow_mut().label = "b".to_string();
    c.0.borrow_mut().label = "c".to_string();
    d.0.borrow_mut().label = "d".to_string();
    e.0.borrow_mut().label = "e".to_string();

    e.backward(); // e = a^2 * (b + c)
    println!("{} {} {} {} {}", a, b, c, d, e);

    let f = Value::new(2.0);
    println!("{}", Value::exp(&f));

    let g = Value::new(2.0);
    let h = Value::new(5.0);

    let i = Value::div(&g, &h);
    i.backward();
    println!("{} {} {}", g, h, i);

    // real neural network
    println!("real nn stuff");
    let mlp = MLP::new(&vec![3, 4, 4, 1]);

    // defining data and labels
    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],];
    let ys = vec![Value::new(1.0), Value::new(-1.0), Value::new(-1.0), Value::new(1.0)];

    // testing initial prediction (spoiler: it's bad)
    let ypred = xs.iter().map(|x| mlp.forward(x)).collect::<Vec<Vec<Value>>>();
    for y in ypred.iter() {
        println!("{}", y[0]);
    }

    // training
    let max_epoch = 100;
    let lr = 0.1;
    for epoch in 0..max_epoch {
        // forward pass
        let ypred = xs.iter().map(|x| mlp.forward(x)[0].clone_rc()).collect::<Vec<Value>>();

        // calculating the loss, specifically MSE
        let mut loss = Value::new(0.0);
        for i in 0..ypred.len() {
            loss = Value::add(&loss, &Value::pow(&Value::sub(&ypred[i], &ys[i]), 2.0));
        }

        // backward pass
        mlp.zero_grad();
        loss.backward();

        // update using gradient descent
        for p in mlp.parameters() {
            p.update_data(-lr * p.get_grad());
        }

        println!("epoch: {} loss: {}", epoch, loss.get_data());
    }

    let ypred = xs.iter().map(|x| mlp.forward(x)[0].clone_rc()).collect::<Vec<Value>>();
    for y in ypred.iter() {
        println!("{}", y);
    }
}
