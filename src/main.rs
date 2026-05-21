use dsp_toy_lib::signal::Signal;

fn main() {
    let mut signal = Signal::from(
        [0, 1, 0, 1]
    );
    let a = Signal::from([5, 5, 5, 5]);
    signal += &a;
    let _ = a.radix_2_fft();
    println!("{}", signal);
}