
pub trait LossFunction {
    fn calculate_loss(&self, output: &Vec<f32>, target: &Vec<f32>) -> f32;
    fn calculate_loss_derivative(&self, output: &Vec<f32>, target: &Vec<f32>) -> Vec<f32>;
}

pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn calculate_loss(&self, output: &Vec<f32>, target: &Vec<f32>) -> f32 {
        let mut loss = 0.0;
        for i in 0..output.len() {
            loss += (output[i] - target[i]).powi(2);
        }
        loss / (output.len() as f32)
    }

    fn calculate_loss_derivative(&self, output: &Vec<f32>, target: &Vec<f32>) -> Vec<f32> {
        let mut loss_derivative = Vec::new();
        for i in 0..output.len() {
            loss_derivative.push(2.0 * (output[i] - target[i]));
        }
        loss_derivative
    }
}