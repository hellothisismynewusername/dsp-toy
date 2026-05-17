use dsp_toy_lib::signal::Signal;

fn main() {
    let signal = Signal::from(
        [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.].as_slice()
    );
    println!("signal:\t\t{:+}", signal);
    
    let window_size = 16;
    let hop_size = 8;

    let mut windows = signal.windows(window_size, hop_size, 2, Signal::hann_window, false).unwrap();

    let a = windows.pop_front().unwrap();
    let b = windows.pop_front().unwrap();

    println!("a_dft:\t\t{:+}\nb_dft:\t\t{:+}", a.radix_2_fft_new().unwrap(), b.radix_2_fft_new().unwrap());
    println!("reconstructed:\t{:+}", a.overlap(&b, hop_size));
}