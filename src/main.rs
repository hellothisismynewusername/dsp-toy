use dsp_toy_lib::signal::Signal;

fn main() {
    let signal = Signal::from(
        [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.].as_slice()
    );
    println!("signal:\t\t{:+}", signal);
    
    let window_size = 16;
    let hop_size = 8;

    let mut windows = signal.windows(window_size, hop_size, 2, Signal::hann_window, false).unwrap();

    println!("a_dft:\t\t{:+}\nb_dft:\t\t{:+}", windows[0].radix_2_fft().unwrap(), windows[1].radix_2_fft().unwrap());

    let a = windows.pop_front().unwrap();
    println!("reconstructed:\t{:+}", a.overlap(&windows[0], hop_size));
}