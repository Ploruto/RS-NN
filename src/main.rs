mod data_types;

use crate::data_types::layers::{self, fully_connected::FullyConnected, layer_types::LayerType};
use crate::data_types::network::neural_network::NeuralNetwork;
fn main() {
    let nn: NeuralNetwork = NeuralNetwork::new()
    .add_input_layer(2)
    .add_fully_connected_layer(50)
    .add_fully_connected_layer(25)
    .add_output_layer(1);

    println!("{:?}", nn);

}
