use crate::layers::layer_types::{Layer, LayerType};
use crate::data_types::functions::activation_functions::ActivationFunction;

#[derive(Clone)]
pub struct OutputLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    prev_layer: Option<Box<dyn Layer>>,
    next_layer: Option<Box<dyn Layer>>,
    activation_function: Option<Box<dyn ActivationFunction>>,
    size: u128, // number of neurons in this layer
}

impl OutputLayer {
    pub fn new(size: u128) -> OutputLayer {
        OutputLayer {
            weights: Vec::new(),
            biases: Vec::new(),
            prev_layer: None,
            next_layer: None,
            activation_function: None,
            size,
        }
    }
}

impl Layer for OutputLayer {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Output
    }

    fn get_weights(&self) -> &Vec<Vec<f32>> {
        &self.weights
    }

    fn get_biases(&self) -> &Vec<f32> {
        &self.biases
    }

    fn get_prev_layer(&self) -> &Option<Box<dyn Layer>> {
        &self.prev_layer
    }

    fn get_next_layer(&self) -> &Option<Box<dyn Layer>> {
        &self.next_layer
    }

    fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }

    fn set_biases(&mut self, biases: Vec<f32>) {
        self.biases = biases;
    }

    fn set_prev_layer(&mut self, prev_layer: Option<Box<dyn Layer>>) {
        self.prev_layer = prev_layer;
    }

    fn set_next_layer(&mut self, next_layer: Option<Box<dyn Layer>>) {
        self.next_layer = next_layer;
    }

    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..self.size {
            let mut sum = 0.0;
            for j in 0..input.len() {
                sum += input[j] * self.weights[j][i as usize];
            }
            sum += self.biases[i as usize];
            output.push(sum);
        }
        output
    }

    fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..self.size {
            let mut sum = 0.0;
            for j in 0..input.len() {
                sum += input[j] * self.weights[i as usize][j];
            }
            sum += self.biases[i as usize];
            output.push(sum);
        }
        output
    }

    fn get_size(&self) -> u128 {
        self.size
    }

    fn get_weights_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.weights
    }

    fn get_biases_mut(&mut self) -> &mut Vec<f32> {
        &mut self.biases
    }

    fn get_activation_function(&self) -> &Option<Box<dyn ActivationFunction>> {
        &self.activation_function
    }

    fn set_activation_function(&mut self, activation_function: Option<Box<dyn ActivationFunction>>) {
        self.activation_function = activation_function;
    }

}