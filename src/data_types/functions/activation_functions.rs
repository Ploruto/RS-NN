use dyn_clone::DynClone;
pub trait ActivationFunction: DynClone {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32>;
    fn backward(&self, input: &Vec<f32>) -> Vec<f32>;
}

dyn_clone::clone_trait_object!(ActivationFunction);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sigmoid;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReLU;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tanh;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Softmax;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Linear;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LeakyReLU;


impl ActivationFunction for Sigmoid {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(1.0 / (1.0 + (-input[i]).exp()));
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i] * (1.0 - input[i]));
        }
        output
    }
}

impl ActivationFunction for ReLU {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i].max(0.0));
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(if input[i] > 0.0 { 1.0 } else { 0.0 });
        }
        output
    }
}

impl ActivationFunction for Tanh {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i].tanh());
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(1.0 - input[i].powi(2));
        }
        output
    }
}

impl ActivationFunction for Softmax {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += input[i].exp();
        }
        for i in 0..input.len() {
            output.push(input[i].exp() / sum);
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i] * (1.0 - input[i]));
        }
        output
    }
}

impl ActivationFunction for Linear {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.clone()
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for _ in 0..input.len() {
            output.push(1.0);
        }
        output
    }
}

impl ActivationFunction for LeakyReLU {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i].max(0.01 * input[i]));
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(if input[i] > 0.0 { 1.0 } else { 0.01 });
        }
        output
    }
}

