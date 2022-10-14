

pub trait Optimizer {
    fn update(&mut self, weights: &mut Vec<f32>, gradients: &Vec<f32>);
}

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, weights: &mut Vec<f32>, gradients: &Vec<f32>) {
        for i in 0..weights.len() {
            weights[i] -= self.learning_rate * gradients[i];
        }
    }
}