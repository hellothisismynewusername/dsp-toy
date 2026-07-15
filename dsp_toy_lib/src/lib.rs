pub mod consts;
pub mod signal;
pub mod utility;
pub mod math;
pub mod real_time;
pub mod cdylib;

#[cfg(test)]
mod kalman_tests {
    use litemap::LiteMap;
    use nalgebra::{Complex, ComplexField, SMatrix};

    use crate::{real_time::{filters::kalman::{kalman_input::KalmanInput, kalman_linear::FilterKalmanLinear, kalman_linear_complex::FilterKalmanLinearComplex}, real_time_signal_processer::RealTimeSignalProcessor}, utility::{SMatrixTimes, equality_accuracy, round_to_place}};

    #[test]
    /// Values for this example were taken from https://kalmanfilter.net/kalman1d_pn.html#:~:text=EXAMPLE%206%20%E2%80%93%20ESTIMATING%20THE%20TEMPERATURE%20OF%20THE%20LIQUID%20IN%20A%20TANK
    fn kalman_linear_simple_1d_test() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        let mut filter = FilterKalmanLinear::<f64, 1, 1, 0> {
            control: None,
            state_vector: SMatrix::<f64, 1, 1>::new(1.),
            estimate_covariance: SMatrix::<f64, 1, 1>::new(10000.), // error = 100
            measure_covariance: SMatrix::<f64, 1, 1,>::new(0.01), // measurement error = 0.1
            state_transition: SMatrixTimes::<f64, 1, 1>::new(SMatrix::<f64, 1, 1>::new(1.), 0), // modelling a constant value
            process_noise_covariance: Some(SMatrixTimes::<f64, 1, 1>::new(SMatrix::<f64, 1, 1>::new(0.0001), 0)),
            observation: SMatrix::<f64, 1, 1>::new(1.)
        };
        // this example doesn't rely on time step, so we're not using delta_time
        filter.init(&KalmanInput {
            measurement_vector: SMatrix::<f64, 1, 1>::new(1.),
            control_vector: None,
            process_noise_covariance: None,
            delta_time: None
        });

        let mut final_val = -1.;

        let true_values = [50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50., 49.997];
        let measurements = [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965, 50.114];
        for i in 0..10 {
            let inp = KalmanInput {
                measurement_vector: SMatrix::<f64, 1, 1>::new(measurements[i]),
                control_vector: None,
                process_noise_covariance: None,
                delta_time: None
            };
            let tmp = filter.process_sample(&inp);
            if i == 9 {
                final_val = tmp[0];
            }
            if print {
                println!("true:\t{}\nmeas:\t{}\nkalm:\t{}", true_values[i], measurements[i], tmp[0]);
            }
        }

        assert_eq!(50.00, round_to_place(final_val, 2));
    }

    /// Multivariate Linear Kalman test: A drone that captures its x and y position periodically and is controlled.
    /// 
    /// State vector: `[x_pos, y_pos, x_vel, y_vel]`
    /// 
    /// Measure: `[x_pos, y_pos]`
    /// 
    /// Measure noise: Some variance in the camera whose noise axes are independent of one another.
    /// 
    /// Process noise: Affected by random wind accelerations  whose axes are independent from one another.
    /// It's a discrete white noise acceleration model, so it just assumes a sampled acceleration stays constant through its time interval.
    /// 
    /// Control input: Commanded x and y accelerations. Inputs are constant in their time interval.
    //
    // True values:
    // [10.0000,  5.0000,  0.0000,  0.0000]
    // [10.5001,  5.0299,  1.0002,  0.0597]
    // [11.9730,  5.0006,  1.9454, -0.1184]
    // [14.1229,  4.9330,  2.3545, -0.0167]
    // [16.4834,  5.4504,  2.3665,  1.0513]
    // [18.8007,  6.8397,  2.2681,  1.7272]
    // [20.8678,  8.8026,  1.8660,  2.1986]
    // [22.2443, 10.9082,  0.8871,  2.0125]
    // [22.6285, 12.7402, -0.1187,  1.6516]
    // [22.3754, 14.0961, -0.3876,  1.0601]
    // [21.7977, 15.0272, -0.7678,  0.8022]
    #[test]
    fn kalman_linear_multivariate_test() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        const TIME_STEP : f64 = 0.5;
        const STATE_DIM : usize = 4;
        const MEASURE_DIM : usize = 2;
        const CONTROL_DIM : usize = 2;
        // As standard deviations of each discrete acceleration drawn for a time interval
        const MEASURE_NOISE : f64 = 0.8;
        const PROCESS_NOISE : f64 = 0.2;

        let measure_noise_as_variance : f64 = MEASURE_NOISE.powf(2.);
        let process_noise_as_variance : f64 = PROCESS_NOISE.powf(2.);

        let initial_state_vector = SMatrix::<f64, STATE_DIM, 1>::new(10., 5., 0., 0.);
        // we're not certain at all about the drone's initial velocity, so we'll just use a huge value.
        let initial_estimate_covariances = SMatrix::<f64, STATE_DIM, STATE_DIM>::new(
            measure_noise_as_variance, 0., 0., 0.,
            0., measure_noise_as_variance, 0., 0.,
            0., 0., 10000., 0.,
            0., 0., 0., 10000.
        );

        let map_state_transition : LiteMap<(usize, usize), f64> = LiteMap::from_iter([
            ((0, 2,), 1.),
            ((1, 3,), 1.),
        ]);
        let map_control : LiteMap<(usize, usize), f64> = LiteMap::from_iter([
            ((0, 0), 2.),
            ((1, 1), 2.),
            ((2, 0), 1.),
            ((3, 1), 1.),
        ]);
        let map_process_noise_covariance : LiteMap<(usize, usize), f64> = LiteMap::from_iter([
            ((0, 0,), 4.),
            ((0, 2,), 3.),
            ((1, 1,), 4.),
            ((1, 3,), 3.),
            ((2, 0,), 3.),
            ((2, 2,), 2.),
            ((3, 1,), 3.),
            ((3, 3,), 2.),
        ]);

        let mut filter = FilterKalmanLinear::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
            state_vector: initial_state_vector,
            estimate_covariance: initial_estimate_covariances,
            observation: SMatrix::<f64, MEASURE_DIM, STATE_DIM>::new(1., 0., 0., 0., 0., 1., 0., 0.),
            measure_covariance: SMatrix::<f64, MEASURE_DIM, MEASURE_DIM>::new(measure_noise_as_variance, 0., 0., measure_noise_as_variance),

            // dt's are put in place by `map_state_transition`.
            state_transition: SMatrixTimes::<f64, STATE_DIM, STATE_DIM>::new_with_litemap(
                SMatrix::<f64, STATE_DIM, STATE_DIM>::new(
                    1., 0., 1., 0., 
                    0., 1., 0., 1.,
                    0., 0., 1., 0.,
                    0., 0., 0., 1.),
                map_state_transition
            ),
            control: Some(SMatrixTimes::<f64, STATE_DIM, CONTROL_DIM>::new_with_litemap(
        SMatrix::<f64, STATE_DIM, CONTROL_DIM>::new(0.5, 0., 0., 0.5, 1., 0., 0., 1.),
                map_control
            )),
            process_noise_covariance: Some(SMatrixTimes::<f64, STATE_DIM, STATE_DIM>::new_with_litemap(
                SMatrix::<f64, STATE_DIM, STATE_DIM>::new(
                    0.25 * process_noise_as_variance, 0., 0.5 * process_noise_as_variance, 0.,
                    0., 0.25 * process_noise_as_variance, 0., 0.5 * process_noise_as_variance,
                    0.5 * process_noise_as_variance, 0., process_noise_as_variance, 0.,
                    0., 0.5 * process_noise_as_variance, 0., process_noise_as_variance),
                map_process_noise_covariance
            ))
        };

        if print {
            println!("state transition {:?}", filter.state_transition.multiply_entries_float(TIME_STEP));
            println!("control {:?}", filter.control.as_ref().unwrap().multiply_entries_float(TIME_STEP));
            println!("process_noise_covariance {:?}", filter.process_noise_covariance.as_ref().unwrap().multiply_entries_float(TIME_STEP));
        }

        let inputs = [
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(9.1111250, 5.2244625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(10.6186250, 4.8505250),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(9.0173125, 4.5523375),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.5,  0.3)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(11.5820375, 5.2031625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.8)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(10.9760625, 5.0775750),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.8)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(11.9340375, 5.3035000),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-0.5,  0.4)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.9099000, 5.8308875),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.1311625, 7.6425000),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-1.0, -0.5)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(12.6269875, 7.1845625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0, -0.5)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.0378875, 7.5577500),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
        ];

        let mut final_estimate = SMatrix::<f64, STATE_DIM, 1>::zeros();
        for (i, input) in inputs.iter().enumerate() {
            let tmp: SMatrix<f64, STATE_DIM, 1> = filter.process_sample(input);
            if i == inputs.len() - 1 {
                final_estimate = tmp.clone();
            }
            if print {
                println!("Calculated estimate at {}: {:?}", i, tmp);
            }
        }
        assert_eq!(12.98, round_to_place(final_estimate[(0, 0)], 2));
    }

    /// Complex linear Kalman filter test: tracks a rotating complex I/Q phasor.
    ///
    /// State: A single complex value representing the signal phasor, `I + jQ`.
    ///
    /// Measurement: A noisy complex I/Q sample of the phasor.
    ///
    /// Measurement noise: Proper, circularly symmetric complex Gaussian noise. Its real and imaginary components are independent and have equal variance.
    ///
    /// Process noise: Small unpredictable changes in the phasor not captured by the model.
    /// The real and imaginary components are independent and have equal variance; the disturbance has no preferred direction or phase in the I/Q plane.
    ///
    /// Control input: A phasor applied in a sample step.
    ///
    /// The phasor rotates by a known angle between successive samples.
    //
    // True values:
    // 1.000000000 + 0.000000000j,
    // 0.926025404 + 0.500000000j,
    // Complex64::new( 0.546961524,  0.904025404),
    // Complex64::new( 0.041669873,  1.036389727),
    // Complex64::new(-0.476107695,  0.922374769),
    // Complex64::new(-0.910508743,  0.580746134),
    // Complex64::new(-1.075896769,  0.042686533),
    // Complex64::new(-0.943097201, -0.464980762),
    // Complex64::new(-0.588255753, -0.878233753),
    // Complex64::new(-0.065327550, -1.052700617),
    // Complex64::new( 0.446774991, -0.943329251),
    #[test]
    fn complex_kalman_linear_test() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        const STATE_DIM: usize = 1;
        const MEASURE_DIM: usize = 1;
        const CONTROL_DIM: usize = 1;

        const PROCESS_NOISE_RMS: f64 = 0.05;
        const MEASUREMENT_NOISE_RMS: f64 = 0.2;

        let theta = std::f64::consts::PI / 6.0;

        let rotation: Complex<f64> = Complex::<f64>::new(theta.cos(), theta.sin());

        let process_variance = PROCESS_NOISE_RMS.powi(2);
        let measurement_variance = MEASUREMENT_NOISE_RMS.powi(2);

        let state_transition =
            SMatrix::<Complex<f64>, STATE_DIM, STATE_DIM>::new(rotation);

        let control =
            SMatrix::<Complex<f64>, STATE_DIM, CONTROL_DIM>::new(Complex::<f64>::new(1.0, 0.0));

        let observation =
            SMatrix::<Complex<f64>, MEASURE_DIM, STATE_DIM>::new(Complex::<f64>::new(1.0, 0.0));

        let process_noise_covariance =
            SMatrix::<Complex<f64>, STATE_DIM, STATE_DIM>::new(Complex::<f64>::new(process_variance, 0.0));

        let measurement_covariance =
            SMatrix::<Complex<f64>, MEASURE_DIM, MEASURE_DIM>::new(Complex::<f64>::new(measurement_variance, 0.0));

        let mut filter = FilterKalmanLinearComplex::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
            state_vector: SMatrix::<Complex<f64>, STATE_DIM, 1>::new(Complex::<f64>::new(1.08, -0.05)),
            observation: observation,
            measure_covariance: measurement_covariance,
            state_transition: SMatrixTimes::<Complex<f64>, STATE_DIM, STATE_DIM>::new(state_transition, 0),
            control: Some(SMatrixTimes::<Complex<f64>, STATE_DIM, CONTROL_DIM>::new(control, 0)),
            process_noise_covariance: Some(SMatrixTimes::<Complex<f64>, STATE_DIM, STATE_DIM>::new(process_noise_covariance, 0)),
            estimate_covariance: SMatrix::<Complex<f64>, STATE_DIM, STATE_DIM>::new(Complex::new(MEASUREMENT_NOISE_RMS.powf(2.), 0.))
        };

        let inputs = [
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.806025404,  0.53)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.05,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.596961524,  0.994025404)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.021669873,  0.926389727)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.02, -0.01))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.376107695,  0.962374769)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.970508743,  0.600746134)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(-0.03,  0.02))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-1.045896769, -0.037313467)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-1.033097201, -0.404980762)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.01,  0.03))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.518255753, -0.868233753)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.105327550, -1.122700617)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.466774991, -0.893329251)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(-0.02,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
        ];

        let mut final_estimate = SMatrix::<Complex<f64>, STATE_DIM, 1>::zeros();
        for (i, input) in inputs.iter().enumerate() {
            let tmp: SMatrix<Complex<f64>, STATE_DIM, 1> = filter.process_sample(input);
            if i == inputs.len() - 1 {
                final_estimate = tmp.clone();
            }
            if print {
                println!("Calculated estimate at {}: {:?}", i, tmp);
            }
        }
        let final_estimate_re = final_estimate[(0, 0)].real();
        let final_estimate_im = final_estimate[(0, 0)].imaginary();
        assert_eq!(0.45, round_to_place(final_estimate_re, 2));
        assert_eq!(-0.95, round_to_place(final_estimate_im, 2));
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use nalgebra::{ComplexField};

    use crate::{math, signal::signal::Signal, utility::{equality_accuracy}};

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

#[cfg(test)]
mod cdylib_tests {
    use crate::{cdylib::{convolve_real, dft, idft, ir2fft, r2fft}, real_time::{cdylib::{Tuple3F64, free_FilterIIRPeakBellF64, new_FilterIIRPeakBellF64, process_sample_FilterIIRPeakBellF64}, filters::filter_iir_peak_bell::FilterIIRPeakBell, real_time_signal_processer::RealTimeSignalProcessor}, signal::cdylib::{free_Signal, free_windows_Signal, imag_data_Signal, len_Signal, new_Signal, new_from_ptr_Signal, real_data_Signal, reconstruct_Signal, resample_length_Signal, resample_ratio_Signal, windows_Signal}};

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
