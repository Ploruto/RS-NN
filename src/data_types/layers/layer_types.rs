use super::super::functions::activation_functions::ActivationFunction;
use dyn_clone::DynClone;


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LayerType {
    FullyConnected,
    Convolutional,
    Pooling,
    Activation,
    Dropout,
    Input,
    Output,
}

pub trait Layer : DynClone {
    fn get_layer_type(&self) -> LayerType;
    fn get_weights(&self) -> &Vec<Vec<f32>>;
    fn get_biases(&self) -> &Vec<f32>;
    fn get_prev_layer(&self) -> &Option<Box<dyn Layer>>;
    fn get_next_layer(&self) -> &Option<Box<dyn Layer>>;
    fn set_weights(&mut self, weights: Vec<Vec<f32>>);
    fn set_biases(&mut self, biases: Vec<f32>);
    fn set_prev_layer(&mut self, prev_layer: Option<Box<dyn Layer>>);
    fn set_next_layer(&mut self, next_layer: Option<Box<dyn Layer>>);
    fn forward(&self, input: &Vec<f32>) -> Vec<f32>;
    fn backward(&self, input: &Vec<f32>) -> Vec<f32>;

    fn get_size(&self) -> u128;

    fn get_weights_mut(&mut self) -> &mut Vec<Vec<f32>>;
    fn get_biases_mut(&mut self) -> &mut Vec<f32>;

    fn get_activation_function(&self) -> &Option<Box<dyn ActivationFunction>>;
    fn set_activation_function(&mut self, activation_function: Option<Box<dyn ActivationFunction>>);
}


dyn_clone::clone_trait_object!(Layer);