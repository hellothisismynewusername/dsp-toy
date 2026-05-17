pub mod consts;
pub mod signal;
pub mod utility;
pub mod math;

#[cfg(test)]
mod tests {
    use std::ops::Mul;

    use crate::signal::Signal;

    #[test]
    fn amplitude_change() {
        let signal = Signal::from([0.1, 0.2, -6., -7.]);
        println!("signal:\t\t\t{}", signal);
        println!("signal amplified:\t{}", signal.clone() * -2.5);

        assert_eq!(signal * -2.5, Signal::from([-0.25, -0.5, 15., 17.5]));
    }

    #[test]
    fn windowing() {
        let signal = Signal::from(
            [10].repeat(40).as_slice()
        );
        println!("signal:\t\t{}", signal);
        
        let window_size = 20;
        let hop_size = 10;

        let mut windows = signal.windows(window_size, hop_size, 3, Signal::hann_window, false).unwrap();

        let a = windows.pop_front().unwrap();
        let signal_reconstructed = a.overlap(&windows[0], hop_size).overlap(&windows[1], hop_size * 2);
        println!("recons:\t\t{}", signal_reconstructed);

        assert_eq!(signal_reconstructed, Signal::from([0., 0.245, 0.955, 2.061, 3.455, 5., 6.545, 7.939, 9.045, 9.755, 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.755, 9.045, 7.939, 6.545, 5., 3.455, 2.061, 0.955, 0.245]));
    }

    #[test]
    fn stft() {
        let signal = Signal::from(
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.]
        );
        println!("signal:\t\t{:+}", signal);
        
        let window_size = 16;
        let hop_size = 8;

        let mut windows = signal.windows(window_size, hop_size, 2, Signal::hann_window, false).unwrap();

        let a = windows.pop_front().unwrap(); // mainly first part of the signal
        let b = windows.pop_front().unwrap(); // mainly second part of the signal
        // of course, a good amount of leakage is present

        println!("a_dft:\t\t{:+}\nb_dft:\t\t{:+}", a.radix_2_fft_new().unwrap(), b.radix_2_fft_new().unwrap());

        println!("reconstructed:\t{:+}", a.clone().overlap(&b, hop_size));

        // bins at which fundamental frequencies land in
        let a_fft_nyquist_bin = a.radix_2_fft_new().unwrap().data[8];
        let b_fft_bin_4 = b.radix_2_fft_new().unwrap().data[4];
        let b_fft_bin_12 = b.radix_2_fft_new().unwrap().data[12]; // b bin 4's corresponding negative frequency

        // assert that the magnitude at the nyquist bin (which is the bin that holds the fundamental) in a is the loudest
        assert!(a.data.iter().all(|x| a_fft_nyquist_bin.real() >= x.real()));
        // assert the equivalence of b's fundamental and its corresponding negative version
        assert_eq!(b_fft_bin_4, b_fft_bin_12.conjugate());
        // assert that the magnitude of b's fundamental is the loudest
        assert!(b.data.iter().all(|x| b_fft_bin_4.real() >= x.real()));
    }

    #[test]
    fn dft_fft_and_inverses_work() {
        let signal = Signal::from(
            [0.1, 0., -1., 0., 1.5, -10.3, 0., 0., 0., 15., 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]
        );

        let dft = signal.forward_dft();
        let dft_idft = dft.inverse_dft();
        let fft = signal.radix_2_fft_new().unwrap();
        let fft_ifft = fft.inverse_radix_2_fft_new().unwrap();
        let dft_ifft = dft.inverse_radix_2_fft_new().unwrap();
        let fft_idft = fft.inverse_dft();

        println!("signal:\t\t{}", signal);
        println!("signal dft:\t\t{}", dft);
        println!("signal fft:\t\t{}", fft);
        println!("signal dft_idft:\t\t{}", dft_idft);
        println!("signal fft_ifft:\t\t{}", fft_ifft);
        println!("signal dft_ifft:\t\t{}", dft_ifft);
        println!("signal fft_idft:\t\t{}", fft_idft);

        assert_eq!(dft, fft);
        assert_eq!(dft_idft, signal);
        assert_eq!(fft_ifft, signal);
        assert_eq!(dft_idft, fft_ifft);
        assert_eq!(dft_ifft, fft_idft);
    }

    #[test]
    fn differentiation_filter() {
        let signal = Signal::from(
            [0, 5, 10, 3, 3, 3, 3, -3, -3, 0, 0]
        );
        let filter = Signal::from(
            [1., -1.]
        ).zero_extend_end(9);

        let differentiated_signal = signal.forward_dft().mul(&filter.forward_dft()).inverse_dft();
        println!("signal: {}\tir: {}", signal, filter);
        println!("differentiated: {:#}", differentiated_signal);

        // 1-sample offset
        assert_eq!(differentiated_signal, Signal::from([0, 5, 5, -7, 0, 0, 0, -6, 0, 3, 0]));
    }

    #[test]
    fn convolve_with_impulse_is_unchanged() {
        let signal = Signal::from(
            [1, 0, -1, 0]
        );
        let ir = Signal::from(
            [1, 0, 0, 0]
        );
        
        println!("signal: {}, ir: {}", signal, ir);

        let ir_dft = ir.forward_dft();

        // Convolution by doing multiplication in the frequency domain
        let signal_final = signal.forward_dft().mul(&ir_dft).inverse_dft();

        println!("signal final {}", signal_final);

        assert_eq!(signal, signal_final)
    }

    #[test]
    fn inverse() {
        let s = Signal::from([0, 1, 0, -1]);
        println!("s:\t\t{}\ns_dft:\t\t{}\ns_dft_idft:\t{}", s, s.forward_dft(), s.forward_dft().inverse_dft());
        assert_eq!(s, s.forward_dft().inverse_dft());
    }

    #[test]
    fn linearity() {
        let signal = Signal::from([1, 0, -1, 0]);
        let dc = Signal::from([1, 1, 1, 1]);

        let mut signal_dft = signal.forward_dft();
        let dc_dft = dc.forward_dft();

        let signal_plus_dc = Signal::from([2, 1, 0, 1]);

        let combine_dft = signal_plus_dc.forward_dft();

        println!("combine_dft {}", combine_dft);
        println!("cos_dft + &dc_dft {}", &mut signal_dft + &dc_dft);

        assert!(combine_dft == signal_dft);
    }
}