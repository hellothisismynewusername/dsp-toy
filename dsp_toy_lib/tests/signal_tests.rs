#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use nalgebra::{ComplexField};

    use dsp_toy_lib::{math, signal::signal::Signal, utility::{equality_accuracy}};


    #[test]
    fn cepstral_test() {
        assert_eq!(equality_accuracy(), 2);

        let signal_1 = Signal::from([1.00000000, 0.01612721, 0.00003034, 0.00005161, 0.02522079, 0.62565284, 0.01061752, 0.00001772, 0.00008771, 0.03770744, 0.49528636, 0.00695944, 0.00001030, 0.00014835, 0.05425479, 0.41113093, 0.00449563, 0.00000602, 0.00024838, 0.07612987, 0.34464726, 0.00284705, 0.00000365, 0.00041046, 0.10567374, 0.28542572, 0.00176719, 0.00000248, 0.00066986, 0.14809176, 0.22625539, 0.00108415, 0.00000212, 0.00108415, 0.22625539, 0.14809176, 0.00066986, 0.00000248, 0.00176719, 0.28542572, 0.10567374, 0.00041046, 0.00000365, 0.00284705, 0.34464726, 0.07612987, 0.00024838, 0.00000602, 0.00449563, 0.41113093, 0.05425479, 0.00014835, 0.00001030, 0.00695944, 0.49528636, 0.03770744, 0.00008771, 0.00001772, 0.01061752, 0.62565284, 0.02522079, 0.00005161, 0.00003034, 0.01612721]);

        let epsilon = 0.00000001;
        let c1 = signal_1.clone().real_cepstrum(epsilon);
        let c2 = signal_1.clone().power_cepstrum(epsilon);

        assert_eq!(c1, Signal::from([-0.539, 0., -0., -0., -0., 0.5, 0., 0., 0., 0., 0.25, -0., 0., 0., 0., 0.167, 0., 0., 0., 0., 0.125, 0., -0., -0., 0., 0.1, 0., 0., 0., -0., 0.083, -0., -0., -0., 0.083, -0., 0., 0., 0., 0.1, 0., -0., -0., 0., 0.125, 0., 0., 0., 0., 0.167, 0., 0., 0., -0., 0.25, 0., 0., 0., 0., 0.5, -0., -0., -0., 0.]));
        assert_eq!(c2, Signal::from([1.161, 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.25, 0., 0., 0., 0., 0.111, 0., 0., 0., 0., 0.063, 0., 0., 0., 0., 0.04, 0., 0., 0., 0., 0.028, 0., 0., 0., 0.028, 0., 0., 0., 0., 0.04, 0., 0., 0., 0., 0.063, 0., 0., 0., 0., 0.111, 0., 0., 0., 0., 0.25, 0., 0., 0., 0., 1., 0., 0., 0., 0.]));
    }

    #[test]
    fn iir_eq_filter() {
        assert_eq!(equality_accuracy(), 2);

        // random noise
        let signal = Signal::from(
            [-3.2, 4.7, 1.8, -0.5, -4.1, 3.4, 0.9, -2.8, 2.1, -1.6, 0.3, -4.8, 4.0, 1.2, -3.7, 2.6, -0.1, 3.8, -1.3, 4.5]
        );
        let sample_rate = 20;

        let band_1 = (5., 20., 5.);
        let band_2 = (9.9, -20., 0.5);

        let signal_filtered = signal.iir_filter_peak_bell_real([band_1, band_2].as_slice(), sample_rate);

        assert_eq!(signal_filtered, Signal::from([-3.2, 4.16, 4.41, -3.03, -6.06, 4.38, 6.36, -6.98, -2.02, 2.76, 3.37, -8.04, -0.15, 7.96, -3.19, -3.04, 0.47, 8.1, -1., -2.04]));
    }

    #[test]
    fn stft() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.]
        );
        let window_size = 8;
        let hop_size = 4;
        let windows_num = 5;

        let mut windows = signal.windows(window_size, hop_size, windows_num, math::hann, false);
        let slices = windows.make_contiguous().iter_mut().into_slice();

        assert_eq!(Signal::reconstruct(slices, hop_size), Signal::from([0., -0.07, 0.25, -0.43, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -0.5, 0.]));
    }

    #[test]
    fn stft2() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.]
        );
        
        let window_size = 16;
        let hop_size = 8;

        let mut windows = signal.windows(window_size, hop_size, 2, math::hann, false);

        let a = windows.pop_front().unwrap(); // mainly first part of the signal
        let b = windows.pop_front().unwrap(); // mainly second part of the signal
        // of course, a good amount of leakage is present

        // bins at which fundamental frequencies land in
        let a_fft_nyquist_bin = a.radix_2_fft_new().unwrap()[8];
        let b_fft_bin_4 = b.radix_2_fft_new().unwrap()[4];
        let b_fft_bin_12 = b.radix_2_fft_new().unwrap()[12]; // b bin 4's corresponding negative frequency

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
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from([1., 0., -1., 0.]).zero_extend_end(4);
        let ir = Signal::from([0.8, 0.3, -1000., 0.]).zero_extend_end(4);
        let convolved_standard = signal.clone().convolve(&ir);
        let convolved_through_mult = signal.forward_dft().mul(&ir.forward_dft()).inverse_dft();

        assert_eq!(convolved_standard.crop(0, 8), convolved_through_mult);
    }

    #[test]
    /// A linear combination of signals is the same linear combination of their respective transforms.
    fn linearity() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from([1., 0., -1., 0.]);
        let dc = Signal::from([1., 1., 1., 1.]);

        let signal_dft = signal.forward_dft();
        let dc_dft = dc.forward_dft();

        let signal_plus_dc = signal + &dc;

        let combine_dft = signal_plus_dc.forward_dft(); 

        assert!(combine_dft == signal_dft + &dc_dft);
    }

    #[test]
    /// Resampling by converting to frequency domain and adding / removing high frequencies.
    fn resample() {
        assert_eq!(equality_accuracy(), 2);

        let s = Signal::from([1., 0., -1., 0., 1., 0., -1., 0., 0., 0., 0., 0.]);
        let s_resampled = s.resample(None, Some(13)).unwrap();

        let s2 = Signal::from([2., 1., 0., 1., 2., 1., 0., 1., 1., 1., 1., 1.]);
        let s2_resampled = s2.resample(Some(1.5), None).unwrap();

        let s3 = Signal::from([0., 1., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0.]);
        let s3_resampled = s3.resample(None, Some(7)).unwrap();

        let s4 = Signal::from([1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1., 0.]);
        let s4_resampled = s4.resample(None, Some(8)).unwrap();

        assert_eq!(s_resampled, Signal::from([1., 0.14, -0.99, -0.32, 0.85, 0.62, -0.82, -0.62, 0.18, -0.11, 0.08, -0.07, 0.07]));
        assert_eq!(s2_resampled, Signal::from([2., 1.56, 0.45, 0., 0.54, 1.46, 2., 1.55, 0.44, 0., 0.64, 1.18, 1., 0.89, 1.11, 1., 0.82, 1.36]));
        assert_eq!(s3_resampled, Signal::from([0.42, 0.23, -0.61, 0.78, -0.2, 0.17, -0.21]));
        assert_eq!(s4_resampled, Signal::from([1., -0.71, 0., 0.71, -1., 0.71, 0., -0.71]));
    }

    #[test]
    fn dft_fft_and_inverses_work() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [0.1, 0., -1., 0., 1.5, -10.3, 0., 0., 0., 15., 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]
        );

        let dft = signal.forward_dft();
        let dft_idft = dft.inverse_dft();
        let fft = signal.radix_2_fft_new().unwrap();
        let fft_ifft = fft.inverse_radix_2_fft_new().unwrap();
        let dft_ifft = dft.inverse_radix_2_fft_new().unwrap();
        let fft_idft = fft.inverse_dft();

        assert_eq!(dft, fft);
        assert_eq!(dft_idft, signal);
        assert_eq!(fft_ifft, signal);
        assert_eq!(dft_idft, fft_ifft);
        assert_eq!(dft_ifft, fft_idft);
    }

    #[test]
    fn differentiation_filter() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [0., 5., 10., 3., 3., 3., 3., -3., -3., 0., 0.]
        );
        let filter = Signal::from(
            [1., -1.]
        ).zero_extend_end(9);

        let differentiated_signal = signal.forward_dft().mul(&filter.forward_dft()).inverse_dft();

        // 1-sample offset
        assert_eq!(differentiated_signal, Signal::from([0., 5., 5., -7., 0., 0., 0., -6., 0., 3., 0.]));
    }

    #[test]
    fn convolve_with_unit_impulse_is_unchanged() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [1., 0., -1., 0.]
        );
        let ir = Signal::from(
            [1., 0., 0., 0.]
        );

        let ir_dft = ir.forward_dft();

        // Convolution by doing multiplication in the frequency domain
        let signal_final = signal.forward_dft().mul(&ir_dft).inverse_dft();

        assert_eq!(signal, signal_final)
    }

    #[test]
    fn windowing() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from(
            [10.].repeat(40).as_slice()
        );
        
        let window_size = 20;
        let hop_size = 10;

        let mut windows = signal.windows(window_size, hop_size, 3, math::hann, false);

        let a = windows.pop_front().unwrap();
        let signal_reconstructed = a.overlap(&windows[0], hop_size).overlap(&windows[1], hop_size * 2);

        assert_eq!(signal_reconstructed, Signal::from([0., 0.24, 0.95, 2.06, 3.45, 5., 6.55, 7.94, 9.05, 9.76, 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.76, 9.05, 7.94, 6.55, 5., 3.45, 2.06, 0.95, 0.24]));
    }

    #[test]
    fn amplitude_change() {
        assert_eq!(equality_accuracy(), 2);

        let signal = Signal::from([0.1, 0.2, -6., -7.]);

        assert_eq!(signal * -2.5, Signal::from([-0.25, -0.5, 15., 17.5]));
    }
}
