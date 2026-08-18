/// Trait for real-time signal processing, in a callback setting in which samples are individually pulled.
/// I is the input type (f64 for audio filter, Complex<f64> for filtering complex signal, KalmanInput for Kalman filter)
pub trait RealTimeSignalProcessor<I, T>{
    fn process_sample(&mut self, inp : I) -> T;
}

/// Trait for real-time signal processing, in a callback setting in which samples are individually pulled.
/// I is the input type (f64 for audio filter, Complex<f64> for filtering complex signal, KalmanInput for Kalman filter).
/// 
/// Some filters (like the KF) have reasonable failures in their implementation such that I decided to have this seperate trait.
pub trait RealTimeSignalProcessorUnreliable<I, T, E>{
    fn process_sample(&mut self, inp : I) -> Result<T, E>;
}