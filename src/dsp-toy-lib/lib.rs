pub mod consts;
pub mod signal;
pub mod utility;
pub mod math;

#[cfg(test)]
mod tests {
    use std::ops::Mul;

    use easy_complex::Complex64;

use crate::signal::{self, Signal};

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
    /// The convolution operation is the same as element-wise multiplication in the other domain.
    fn convolution_theorem() {
        let signal = Signal::from([1, 0, -1, 0]).zero_extend_end(4);
        let ir = Signal::from([0.8, 0.3, -1000., 0.]).zero_extend_end(4);
        let convolved_standard = signal.clone().convolve(&ir);
        let convolved_through_mult = signal.forward_dft().mul(&ir.forward_dft()).inverse_dft();

        println!("convolved_mult:\t\t{:#.2}", convolved_through_mult);
        println!("convolved_stan:\t\t{:#.2}", convolved_standard.crop_new(0, 8));

        assert_eq!(convolved_standard.crop(0, 8), convolved_through_mult);
    }

    #[test]
    /// A linear combination of signals is the same linear combination of their respective transforms.
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

    #[test]
    fn sinc_interpolation_with_freq_domain() {
        let s = Signal::from([0, 1, 0, -1, 0]);
        let mut s_freq = s.forward_dft();

        // Sinc interpolate to a 10 sample signal by inserting zeroes in the middle of the frequency domain.
        for _ in 0..s.len() {
            s_freq.data.insert(s.len() / 2 + 1, Complex64::from(0));
        }
        let interpolated = s_freq.inverse_dft() * 2; // scaling factor is 2, so we need to double the amplitude

        println!("signal:\t\t{:#.2}", s);
        println!("signal_freq:\t{:.2}", s.forward_dft());
        println!("signal_freq_0s:\t{:.2}", s_freq);
        println!("interpolate:\t{:#.2}", interpolated);

        assert_eq!(interpolated, Signal::from([0., 0.45, 1., 0.89, 0., -0.89, -1., -0.45, 0., 0.]));
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
    fn convolve_with_unit_impulse_is_unchanged() {
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
    fn windowing() {
        let signal = Signal::from(
            [10].repeat(40).as_slice()
        );
        println!("signal:\t\t{:+.2}", signal);
        
        let window_size = 20;
        let hop_size = 10;

        let mut windows = signal.windows(window_size, hop_size, 3, Signal::hann_window, false).unwrap();

        let a = windows.pop_front().unwrap();
        let signal_reconstructed = a.overlap(&windows[0], hop_size).overlap(&windows[1], hop_size * 2);
        println!("recons:\t\t{:+.2}", signal_reconstructed);

        assert_eq!(signal_reconstructed, Signal::from([0., 0.24, 0.95, 2.06, 3.45, 5., 6.55, 7.94, 9.05, 9.76, 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.76, 9.05, 7.94, 6.55, 5., 3.45, 2.06, 0.95, 0.24]));
    }

    #[test]
    fn amplitude_change() {
        let signal = Signal::from([0.1, 0.2, -6., -7.]);
        println!("signal:\t\t\t{}", signal);
        println!("signal amplified:\t{}", signal.clone() * -2.5);

        assert_eq!(signal * -2.5, Signal::from([-0.25, -0.5, 15., 17.5]));
    }
}
