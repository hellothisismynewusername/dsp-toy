#[cfg(test)]
mod cdylib_tests {
    use dsp_toy_lib::{cdylib::{convolve_real, dft, idft, ir2fft, r2fft}, real_time::{cdylib::{Tuple3F64, free_FilterIIRPeakBellF64, new_FilterIIRPeakBellF64, process_sample_FilterIIRPeakBellF64}, filters::filter_iir_peak_bell::FilterIIRPeakBell, real_time_signal_processer::RealTimeSignalProcessor}, signal::cdylib::{free_Signal, free_windows_Signal, imag_data_Signal, len_Signal, new_Signal, new_from_ptr_Signal, real_data_Signal, reconstruct_Signal, resample_length_Signal, resample_ratio_Signal, windows_Signal}};

    fn assert_f64_roughly_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_f64_slice_roughly_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_f64_roughly_eq(*actual, *expected);
        }
    }

    #[test]
    fn math_cdylib_functions_handle_nulls_and_transform_round_trips() {
        assert!(!convolve_real(std::ptr::null_mut(), 1, [1.].as_ptr(), 1));
        assert!(!convolve_real([1.].as_mut_ptr(), 1, std::ptr::null(), 1));
        assert!(!dft(std::ptr::null_mut(), [0.].as_mut_ptr(), 1));
        assert!(!dft([0.].as_mut_ptr(), std::ptr::null_mut(), 1));
        assert!(!idft(std::ptr::null_mut(), [0.].as_mut_ptr(), 1));
        assert!(!idft([0.].as_mut_ptr(), std::ptr::null_mut(), 1));
        assert!(!r2fft(std::ptr::null_mut(), [0.].as_mut_ptr(), 1));
        assert!(!r2fft([0.].as_mut_ptr(), std::ptr::null_mut(), 1));
        assert!(!ir2fft(std::ptr::null_mut(), [0.].as_mut_ptr(), 1));
        assert!(!ir2fft([0.].as_mut_ptr(), std::ptr::null_mut(), 1));

        let mut convolved = vec![1., 2., 0.];
        assert!(convolve_real(convolved.as_mut_ptr(), 2, [3., 4.].as_ptr(), 2));
        assert_f64_slice_roughly_eq(&convolved, &[3., 10., 8.]);

        let mut real = vec![1., 2., 3., 4.];
        let mut imag = vec![0.5, -0.25, 0.75, -1.];
        let original_real = real.clone();
        let original_imag = imag.clone();
        assert!(dft(real.as_mut_ptr(), imag.as_mut_ptr(), real.len()));
        assert!(idft(real.as_mut_ptr(), imag.as_mut_ptr(), real.len()));
        assert_f64_slice_roughly_eq(&real, &original_real);
        assert_f64_slice_roughly_eq(&imag, &original_imag);

        let mut real = vec![0.25, -1., 2.5, 0., 1.25, -0.5, 3., 4.];
        let mut imag = vec![0.5, 0., -0.25, 1.5, -1., 2., 0.75, -0.5];
        let original_real = real.clone();
        let original_imag = imag.clone();
        assert!(r2fft(real.as_mut_ptr(), imag.as_mut_ptr(), real.len()));
        assert!(ir2fft(real.as_mut_ptr(), imag.as_mut_ptr(), real.len()));
        assert_f64_slice_roughly_eq(&real, &original_real);
        assert_f64_slice_roughly_eq(&imag, &original_imag);
    }

    #[test]
    fn signal_cdylib_functions() {
        assert_eq!(len_Signal(std::ptr::null()), usize::MAX);
        assert!(new_from_ptr_Signal(std::ptr::null(), 3).is_null());
        assert_eq!(resample_ratio_Signal(std::ptr::null_mut(), 2.), -1);

        let empty = new_Signal();
        assert!(!empty.is_null());
        assert_eq!(len_Signal(empty), 0);
        free_Signal(empty);

        let input = [1., 0., -1., 0.];
        let signal = new_from_ptr_Signal(input.as_ptr(), input.len());
        assert!(!signal.is_null());
        assert_eq!(len_Signal(signal), input.len());

        let mut out = vec![0.; input.len()];
        assert!(!real_data_Signal(signal, out.as_mut_ptr(), input.len() - 1));
        assert!(real_data_Signal(signal, out.as_mut_ptr(), out.len()));
        assert_f64_slice_roughly_eq(&out, &input);

        let mut imag = vec![1.; input.len()];
        assert!(imag_data_Signal(signal, imag.as_mut_ptr(), imag.len()));
        assert_f64_slice_roughly_eq(&imag, &[0., 0., 0., 0.]);

        let mut signal_for_ratio = signal;
        assert_eq!(resample_ratio_Signal(&mut signal_for_ratio, 2.), 0);
        assert_eq!(len_Signal(signal_for_ratio), 8);

        let mut signal_for_length = signal_for_ratio;
        assert_eq!(resample_length_Signal(&mut signal_for_length, 4), 0);
        assert_eq!(len_Signal(signal_for_length), 4);
        free_Signal(signal_for_length);
    }

    #[test]
    fn signal_windows_and_reconstruct() {
        let input = [1., 1., 1., 1.];
        let signal = new_from_ptr_Signal(input.as_ptr(), input.len());

        let window_size = 2;
        let hop_size = 2;
        let windows_count = 2;
        let window_function = 2; // rectangular

        let windows = windows_Signal(signal, window_size, hop_size, windows_count, window_function, false);

        let reconstructed = reconstruct_Signal(windows, windows_count, hop_size);

        let out = vec![1.; input.len()];
        assert_f64_slice_roughly_eq(&out, &input);

        free_Signal(signal);
        free_Signal(reconstructed);
        free_windows_Signal(windows, windows_count);
    }

    #[test]
    fn real_time_cdylib_filter_matches_rust_filter() {
        assert!(new_FilterIIRPeakBellF64(std::ptr::null(), 1, 48_000).is_null());
        assert!(new_FilterIIRPeakBellF64([Tuple3F64 { a: 1., b: 1., c: 1. }].as_ptr(), 0, 48_000).is_null());
        assert_eq!(
            process_sample_FilterIIRPeakBellF64(std::ptr::null_mut(), 1.),
            -1. * f64::MAX
        );

        let bands = [
            Tuple3F64 {
                a: 5.,
                b: 20.,
                c: 5.,
            },
            Tuple3F64 {
                a: 9.9,
                b: -20.,
                c: 0.5,
            },
        ];
        let filter = new_FilterIIRPeakBellF64(bands.as_ptr(), bands.len(), 20);
        assert!(!filter.is_null());

        let mut expected_filter = FilterIIRPeakBell::new_real(&[(5., 20., 5.), (9.9, -20., 0.5)], 20);
        let signal = [-3.2, 4.7, 1.8, -0.5, -4.1];
        for sample in signal {
            let actual = process_sample_FilterIIRPeakBellF64(filter, sample);
            let expected = expected_filter.process_sample(sample);
            assert_f64_roughly_eq(actual, expected);
        }

        free_FilterIIRPeakBellF64(filter);
    }

}