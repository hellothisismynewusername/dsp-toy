use dsp_toy_lib::{real_time::{filters::filter_iir_peak_bell::FilterIIRPeakBell, real_time_signal_processer::RealTimeSignalProcessor}, signal::signal::Signal};
use easy_complex::Complex64;
use hound::{WavSpec, WavWriter};
use rand::RngExt;

mod example_filter_noise_with_windows;

fn main() {
    let mut rng = rand::rng();
    let mut rand_fn = |_ : usize| -> Complex64 {
        Complex64::from(rng.random_range::<f64, _>((-0.1)..(0.1)))
    };

    let signal = Signal::from_fn_mut(&mut rand_fn, 0, 204800);
    let sample_rate = 22050;

    
    let band = (5000., -50., 30.);
    let mut filter = FilterIIRPeakBell::<f64>::new_real([band].as_slice(), sample_rate);




    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float
    };

    let mut writer2 = WavWriter::create("triple_tuff.wav", spec).unwrap();

    for s in signal.iter_real() {
        writer2.write_sample(filter.process_sample(s) as f32).unwrap();
    }

    writer2.flush().unwrap();
    writer2.finalize().unwrap();
}