use dsp_toy_lib::{math, signal::signal::{self, Signal}};
use easy_complex::Complex64;
use hound::{WavSpec, WavWriter};
use rand::RngExt;

fn do_the_thing() {
    let mut rng = rand::rng();
    let mut rand_fn = |_ : usize| -> Complex64 {
        Complex64::from(rng.random_range::<f64, _>((-0.1)..(0.1)))
    };

    let signal = Signal::from_fn_mut(&mut rand_fn, 0, 204800);
    let sample_rate = 22050;

    let window_size = 4096;
    let hop_size = 2048;
    let window_count = 99;

    let mut windows = signal.windows(window_size, hop_size, window_count, math::hann, false);
    for (i, window) in windows.iter_mut().enumerate() {
        let band_freq = 11000. * (i as f64 / window_count as f64);
        let band = (band_freq, 50., 30.);

        *window = window.iir_filter_peak_bell_real([band].as_slice(), sample_rate);
    }

    let signal_r = Signal::reconstruct(windows.make_contiguous().iter().as_slice(), hop_size);

    println!("signal:\t\t{:+#.2}, sample rate {}", signal, sample_rate);
    println!("filtered:\t{:+#.2}", signal_r);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float
    };
    let mut writer = WavWriter::create("skibidi1.wav", spec).unwrap();

    for s in signal.iter_real().map(|x| x as f32) {
        writer.write_sample(s).unwrap();
    }

    writer.flush().unwrap();
    writer.finalize().unwrap();

    let mut writer2 = WavWriter::create("skibidi2.wav", spec).unwrap();

    for s in signal_r.iter_real().map(|x| x as f32) {
        writer2.write_sample(s).unwrap();
    }

    writer2.flush().unwrap();
    writer2.finalize().unwrap();
}