use dsp_toy_lib::signal::Signal;

fn main() {
    let signal = Signal::from(
        [0, 1, 0, 1]
    );
    println!("fft {:}", signal.radix_2_fft().unwrap());
}