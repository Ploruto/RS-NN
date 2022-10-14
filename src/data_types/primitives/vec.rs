use dyn_clone::DynClone;
pub trait DotProduct {
    fn dot_product(&self, other: &Self) -> f32;
}

impl DotProduct for Vec<f32> {
    fn dot_product(&self, other: &Self) -> f32 {
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}
