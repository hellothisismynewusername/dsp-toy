/// Trait for real-time signal processing, in a callback setting in which samples are individually pulled.
pub trait RealTimeSignalProcessor<T> {
    fn process_sample(&mut self, inp : T) -> T;
}