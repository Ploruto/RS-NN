use std::fmt::{Formatter, Error};
use core::fmt::Debug;

use super::super::functions::{
    loss_functions::{LossFunction, MeanSquaredError},
    optimizers::{Optimizer, SGD},
};
use crate::data_types::layers::{
    fully_connected::FullyConnected,
    input::InputLayer,
    layer_types::{Layer, LayerType},
    output::OutputLayer,
};

pub struct NeuralNetwork {
    input_layer: Option<Box<dyn Layer>>,
    output_layer: Option<Box<dyn Layer>>,

    layers: Vec<Box<dyn Layer>>,
    loss_function: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,

    last_added_layer: Option<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            input_layer: None,
            output_layer: None,
            layers: Vec::new(),
            loss_function: Box::new(MeanSquaredError),
            optimizer: Box::new(SGD::new(0.01)),
            last_added_layer: None,
        }
    }

    pub fn add_input_layer(self: Self, size: u128) -> Self {
        // create new input layer
        let input_layer = Box::new(InputLayer::new(size));
        Self {
            input_layer: Some(input_layer),
            output_layer: self.output_layer,
            layers: self.layers,
            loss_function: self.loss_function,
            optimizer: self.optimizer,
            last_added_layer: self.last_added_layer,
        }
    }

    pub fn add_fully_connected_layer(self: Self, size: u128) -> Self {
        // create new fully connected layer
        let mut fully_connected_layer = Box::new(FullyConnected::new(size));
        // set the previous layer of the new layer to the last added layer
        fully_connected_layer.set_prev_layer(self.last_added_layer.clone());
        // set the next layer of the last added layer to the new layer
        match self.last_added_layer {
            Some(mut layer) => layer.set_next_layer(Some(fully_connected_layer.clone())),
            None => (),
        }
        // add the new layer to the list of layers
        let mut layers = self.layers.clone();
        layers.push(fully_connected_layer.clone());

        // set the last added layer to the new layer
        Self {
            input_layer: self.input_layer,
            output_layer: self.output_layer,
            layers: layers,
            loss_function: self.loss_function,
            optimizer: self.optimizer,
            last_added_layer: Some(fully_connected_layer),
        }
    }

    pub fn add_output_layer(self: Self, size: u128) -> Self {
        // create new output layer
        let mut output_layer = Box::new(OutputLayer::new(size));
        // set the previous layer of the new layer to the last added layer
        output_layer.set_prev_layer(self.last_added_layer.clone());
        // set the next layer of the last added layer to the new layer
        match self.last_added_layer.clone() {
            Some(mut layer) => layer.set_next_layer(Some(output_layer.clone())),
            None => (),
        }

        // set the last added layer to the new layer
        Self {
            input_layer: self.input_layer,
            output_layer: Some(output_layer),
            layers: self.layers,
            loss_function: self.loss_function,
            optimizer: self.optimizer,
            last_added_layer: self.last_added_layer.clone(),
        }
    }


    
}

impl Debug for NeuralNetwork {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        // check if the options are set and unwrap them and write! the size
        match &self.input_layer {
            Some(input_layer) => write!(f, "Input Layer: {} ", input_layer.get_size()),
            None => write!(f, "Input Layer: None "),
        }?;
        match &self.output_layer {
            Some(output_layer) => write!(f, "Output Layer: {} ", output_layer.get_size()),
            None => write!(f, "Output Layer: None "),
        }?;
        write!(f, "Layers: ")?;
        for layer in &self.layers {
            write!(f, "{} ", layer.get_size())?;
        }
        Ok(())

    }
    
}