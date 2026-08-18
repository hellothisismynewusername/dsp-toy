#[cfg(test)]
mod kalman_tests {

    use litemap::LiteMap;
    use nalgebra::{Complex, ComplexField, Const, MatrixView, SMatrix, Vector3, Vector6};
    use rand::{SeedableRng, distr::Distribution};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    use dsp_toy_lib::{math::normalize_angle, real_time::{filters::kalman::{kalman_input::KalmanInput, kalman_linear::FilterKalmanLinear, kalman_linear_complex::FilterKalmanLinearComplex, kalman_unscented::FilterKalmanUnscented, sigma_points_functions::julier::Julier}, real_time_signal_processer::{RealTimeSignalProcessor, RealTimeSignalProcessorUnreliable}}, utility::{SMatrixTimes, equality_accuracy, round_to_place}};

    #[test]
    /// Values for this example were taken from https://kalmanfilter.net/kalman1d_pn.html#:~:text=EXAMPLE%206%20%E2%80%93%20ESTIMATING%20THE%20TEMPERATURE%20OF%20THE%20LIQUID%20IN%20A%20TANK
    fn kalman_linear_simple_1d_test() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        let mut filter = FilterKalmanLinear::<f64, f64, 1, 1, 0> {
            control: None,
            state_vector: SMatrix::<f64, 1, 1>::new(1.),
            estimate_covariance: SMatrix::<f64, 1, 1>::new(10000.), // error = 100
            measure_covariance: SMatrix::<f64, 1, 1,>::new(0.01), // measurement error = 0.1
            state_transition: SMatrixTimes::<f64, f64, 1, 1>::new(SMatrix::<f64, 1, 1>::new(1.), 0), // modelling a constant value
            process_noise_covariance: Some(SMatrixTimes::<f64, f64, 1, 1>::new(SMatrix::<f64, 1, 1>::new(0.0001), 0)),
            observation: SMatrix::<f64, 1, 1>::new(1.)
        };
        // this example doesn't rely on time step, so we're not using delta_time
        filter.init(&KalmanInput {
            measurement_vector: SMatrix::<f64, 1, 1>::new(1.),
            control_vector: None,
            process_noise_covariance: None,
            delta_time: None
        }).unwrap();

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
            let tmp = filter.process_sample(&inp).unwrap();
            if i == 9 {
                final_val = tmp[0];
            }
            if print {
                println!("true:\t{}\nmeas:\t{}\nkalm:\t{}", true_values[i], measurements[i], tmp[0]);
            }
        }

        assert_eq!(50.00, round_to_place(final_val, equality_accuracy()));
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

        let mut filter = FilterKalmanLinear::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
            state_vector: initial_state_vector,
            estimate_covariance: initial_estimate_covariances,
            observation: SMatrix::<f64, MEASURE_DIM, STATE_DIM>::new(1., 0., 0., 0., 0., 1., 0., 0.),
            measure_covariance: SMatrix::<f64, MEASURE_DIM, MEASURE_DIM>::new(measure_noise_as_variance, 0., 0., measure_noise_as_variance),

            // dt's are put in place by `map_state_transition`.
            state_transition: SMatrixTimes::<f64, f64, STATE_DIM, STATE_DIM>::new_with_litemap(
                SMatrix::<f64, STATE_DIM, STATE_DIM>::new(
                    1., 0., 1., 0., 
                    0., 1., 0., 1.,
                    0., 0., 1., 0.,
                    0., 0., 0., 1.),
                map_state_transition
            ),
            control: Some(SMatrixTimes::<f64, f64, STATE_DIM, CONTROL_DIM>::new_with_litemap(
        SMatrix::<f64, STATE_DIM, CONTROL_DIM>::new(0.5, 0., 0., 0.5, 1., 0., 0., 1.),
                map_control
            )),
            process_noise_covariance: Some(SMatrixTimes::<f64, f64, STATE_DIM, STATE_DIM>::new_with_litemap(
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
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(9.1111250, 5.2244625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(10.6186250, 4.8505250),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(9.0173125, 4.5523375),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.5,  0.3)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(11.5820375, 5.2031625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.8)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(10.9760625, 5.0775750),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.8)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(11.9340375, 5.3035000),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-0.5,  0.4)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.9099000, 5.8308875),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-1.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.1311625, 7.6425000),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(-1.0, -0.5)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(12.6269875, 7.1845625),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0, -0.5)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
            KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<f64, MEASURE_DIM, 1>::new(13.0378875, 7.5577500),
                control_vector: Some(SMatrix::<f64, CONTROL_DIM, 1>::new(0.0,  0.0)),
                process_noise_covariance: None,
                delta_time: Some(TIME_STEP)
            },
        ];

        let mut final_estimate = SMatrix::<f64, STATE_DIM, 1>::zeros();
        for (i, input) in inputs.iter().enumerate() {
            let tmp: SMatrix<f64, STATE_DIM, 1> = filter.process_sample(input).unwrap();
            if i == inputs.len() - 1 {
                final_estimate = tmp.clone();
            }
            if print {
                println!("Calculated estimate at {}: {:?}", i, tmp);
            }
        }
        assert_eq!(12.98, round_to_place(final_estimate[(0, 0)], equality_accuracy()));
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
    // 0.546961524 + 0.904025404j,
    // 0.041669873 + 1.036389727j,
    // -0.476107695 + 0.922374769j,
    // -0.910508743 + 0.580746134j,
    // -1.075896769 + 0.042686533j,
    // -0.943097201 - 0.464980762j,
    // -0.588255753 - 0.878233753j,
    // -0.065327550 - 1.052700617j,
    // 0.446774991 - 0.943329251j,
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

        let mut filter = FilterKalmanLinearComplex::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
            state_vector: SMatrix::<Complex<f64>, STATE_DIM, 1>::new(Complex::<f64>::new(1.08, -0.05)),
            observation: observation,
            measure_covariance: measurement_covariance,
            state_transition: SMatrixTimes::<Complex<f64>, f64, STATE_DIM, STATE_DIM>::new(state_transition, 0),
            control: Some(SMatrixTimes::<Complex<f64>, f64, STATE_DIM, CONTROL_DIM>::new(control, 0)),
            process_noise_covariance: Some(SMatrixTimes::<Complex<f64>, f64, STATE_DIM, STATE_DIM>::new(process_noise_covariance, 0)),
            estimate_covariance: SMatrix::<Complex<f64>, STATE_DIM, STATE_DIM>::new(Complex::new(MEASUREMENT_NOISE_RMS.powf(2.), 0.))
        };

        let inputs = [
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.806025404,  0.53)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.05,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.596961524,  0.994025404)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(0.021669873,  0.926389727)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.02, -0.01))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.376107695,  0.962374769)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.970508743,  0.600746134)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(-0.03,  0.02))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-1.045896769, -0.037313467)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-1.033097201, -0.404980762)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.01,  0.03))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.518255753, -0.868233753)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                measurement_vector: SMatrix::<Complex<f64>, MEASURE_DIM, 1>::new(Complex::<f64>::new(-0.105327550, -1.122700617)),
                control_vector: Some(SMatrix::<Complex<f64>, CONTROL_DIM, 1>::new(Complex::<f64>::new(0.00,  0.00))),
                process_noise_covariance: None,
                delta_time: None
            },
            KalmanInput::<Complex<f64>, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
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
        assert_eq!(0.45, round_to_place(final_estimate_re, equality_accuracy()));
        assert_eq!(-0.95, round_to_place(final_estimate_im, equality_accuracy()));
    }

    /// Adapted from first run example "Robot Localization - A Fully Worked Example" from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
    /// 
    /// Due to the deterministic random number generation used, the loop is far from "blazingly fast".
    /// 
    /// This is almost entirely not the fault of the filter though; it runs much faster without such rng.
    #[test]
    fn unscented_kalman_test() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        const WHEELBASE : f64 = 0.5;
        const NUM_BEARINGS : usize = 3;

        const STATE_DIM : usize = 3;
        const N_OUT : usize = STATE_DIM * 2 + 1; // aka number of sigma points
        const MEASURE_DIM : usize = NUM_BEARINGS * 2;
        const CONTROL_DIM : usize = 2;
        const TIME_STEP : f64 = 1.; // frequency of running full predict/correct UKF step
        const SIM_STEP : f64 = 0.1; // frequency of updating simulated robot position

        const SIGMA_RANGE : f64 = 0.3;
        const SIGMA_BEARING : f64 = 0.1;
        const MAGIC_NUMBER : f64 = 0.0001; // used in Q

        // Must be of length `NUM_BEARINGS`
        let landmarks = [
            (5., 10.),
            (10., 5.),
            (15., 15.)
        ];

        let init_state_vector =
            SMatrix::<f64, STATE_DIM, 1>::new(2., 6., 0.3);
        let init_estimate_covariance =
            SMatrix::<f64, STATE_DIM, STATE_DIM>::from_diagonal(&Vector3::new(0.1, 0.1, 0.05));
        let measure_covariance =
            SMatrix::<f64, MEASURE_DIM, MEASURE_DIM>::from_diagonal(
                &Vector6::from_column_slice(&[SIGMA_RANGE.powi(2), SIGMA_BEARING.powi(2)].repeat(NUM_BEARINGS))
            );
        let process_noise_covariance = Some(
            SMatrixTimes::<f64, f64, STATE_DIM, STATE_DIM>::new(SMatrix::identity() * MAGIC_NUMBER, 0)
        );

        let state_transition_function = move |state : SMatrix<f64, STATE_DIM, 1>, time_step : f64, control_o : Option<SMatrix<f64, CONTROL_DIM, 1>>| {
            let curr_angle = state[2];
            
            if let Some(control) = control_o {
                let vel = control[0];
                let steer_angle = control[1];
                let dist = vel * time_step; // euler method

                if steer_angle.abs() > 0.001 {
                    let beta = (dist / WHEELBASE) * steer_angle.tan();
                    let turn_radius = WHEELBASE / steer_angle.tan();

                    let (sinh, sinhb) = (curr_angle.sin(), (curr_angle + beta).sin());
                    let (cosh, coshb) = (curr_angle.cos(), (curr_angle + beta).cos());
                    state + SMatrix::<f64, STATE_DIM, 1>::new(
                        turn_radius * sinhb - turn_radius * sinh,
                        turn_radius * cosh - turn_radius * coshb,
                        beta
                    )
                } else {
                    state + SMatrix::<f64, STATE_DIM, 1>::new(
                        dist * curr_angle.cos(),
                        dist * curr_angle.sin(),
                        0.
                    )
                }
            } else {
                panic!("Impossible") // using control for this example
            }
        };

        // measurement
        let residual_h = |a : SMatrix<f64, MEASURE_DIM, 1>, b : SMatrix<f64, MEASURE_DIM, 1>| {
            let mut residual = a - b;
            for i in (0..residual.len()).step_by(2) {
                residual[i + 1] = normalize_angle(residual[i + 1]);
            }
            residual
        };
        // state
        let residual_x = |a : SMatrix<f64, STATE_DIM, 1>, b : SMatrix<f64, STATE_DIM, 1>| {
            let mut residual = a - b;
            residual[2] = normalize_angle(residual[2]);
            residual
        };

        assert_eq!(landmarks.len(), NUM_BEARINGS);

        // take a state variable and return the measurement that would correspond to that state.
        let observation_function = |state : SMatrix<f64, STATE_DIM, 1>| {
            let mut out = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

            for (i, (px, py)) in landmarks.iter().enumerate() {
                let dist = ((px - state[0]).powi(2) + (py - state[1]).powi(2)).sqrt();
                let angle = (py - state[1]).atan2(px - state[0]);
                out[2 * i] = dist;
                out[2 * i + 1] = normalize_angle(angle - state[2]);
            }

            out
        };

        let state_mean_function = |sigmas : [SMatrix<f64, STATE_DIM, 1>; N_OUT], weights : [f64; N_OUT]| {
            let mut out = SMatrix::<f64, STATE_DIM, 1>::zeros();

            let sum_sin = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[2].sin() * weight)
                .fold(0., |acc, x| acc + x);
            let sum_cos = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[2].cos() * weight)
                .fold(0., |acc, x| acc + x);

            out[0] = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[0] * weight)
                .fold(0., |acc, x| acc + x);
            out[1] = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[1] * weight)
                .fold(0., |acc, x| acc + x);
            out[2] = sum_sin.atan2(sum_cos);

            out
        };

        let measure_mean_function = |sigmas : [SMatrix<f64, MEASURE_DIM, 1>; N_OUT], weights : [f64; N_OUT]| {
            let mut out = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

            for i in (0..MEASURE_DIM).step_by(2) {
                let sum_sin = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i + 1].sin() * weight)
                    .fold(0., |acc, x| acc + x);
                let sum_cos = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i + 1].cos() * weight)
                    .fold(0., |acc, x| acc + x);

                out[i] = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i] * weight)
                    .fold(0., |acc, x| acc + x);
                out[i + 1] = sum_sin.atan2(sum_cos);
            }

            out
        };

        let add_state_function = |a : SMatrix<f64, STATE_DIM, 1>, b : MatrixView<'_, f64, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>| {
            SMatrix::<f64, STATE_DIM, 1>::new(
                a[0] + b[0],
                a[1] + b[1],
                normalize_angle(a[2] + b[2])
            )
        };

        let mut filter =
            FilterKalmanUnscented::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, _, _, Julier<f64, STATE_DIM, N_OUT>, _, _, _, _, _> {
                state_vector: init_state_vector,
                estimate_covariance: init_estimate_covariance,
                measure_covariance: measure_covariance,
                process_noise_covariance: process_noise_covariance,
                observation: observation_function,
                state_transition: state_transition_function,
                sigma_generator_function: Julier::new(0.),
                state_mean_function: state_mean_function,
                measure_mean_function: measure_mean_function,
                residual_z_function: residual_h,
                residual_x_function: residual_x,
                add_state_function: add_state_function
        };

        let commands = [
            SMatrix::<f64, CONTROL_DIM, 1>::new(1.1, 0.01)
        ].repeat(200);

        let mut sim_pos = init_state_vector;

        let mut rng = ChaCha8Rng::seed_from_u64(67);

        let mut final_estimate = None;

        // The author of the original first robot example did include a `sigma_steer`, but I think they didn't actually use it. So I didn't either.
        for (i, command) in commands.iter().enumerate() {
            sim_pos = state_transition_function(sim_pos, SIM_STEP, Some(*command));

            if print { println!("i = {}\tposition:\t{:?}", i, sim_pos); }

            if (i + 1) % (TIME_STEP / SIM_STEP) as usize == 0 {
                let mut measurement_vector = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

                for (j, landmark) in landmarks.iter().enumerate() {
                    let (dx, dy) = (landmark.0 - sim_pos[0], landmark.1 - sim_pos[1]);

                    let randomness : f64 = StandardNormal.sample(&mut rng);
                    let dist = (dx.powi(2) + dy.powi(2)).sqrt() + randomness * SIGMA_RANGE;

                    let randomness : f64 = StandardNormal.sample(&mut rng);
                    let bearing = (landmark.1 - sim_pos[1]).atan2(landmark.0 - sim_pos[0]);
                    let a = normalize_angle(bearing - sim_pos[2] + randomness * SIGMA_BEARING);

                    measurement_vector[2 * j] = dist;
                    measurement_vector[2 * j + 1] = a;
                }

                let processed = filter.process_sample(&KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                    measurement_vector: measurement_vector,
                    control_vector: Some(*command),
                    process_noise_covariance: None,
                    delta_time: Some(TIME_STEP), // not technically necessary since TIME_STEP = 1.
                });

                if print { println!("i = {}\tPROCESSED:\t{:?}", i, processed); }

                if i == commands.len() - 1 {
                    final_estimate = Some(processed);
                }
            }
        }

        assert!(final_estimate.is_some());
        assert_eq!(20.85, round_to_place(final_estimate.unwrap()[0], equality_accuracy()));
        assert_eq!(16.83, round_to_place(final_estimate.unwrap()[1], equality_accuracy()));
        assert_eq!(0.73, round_to_place(final_estimate.unwrap()[2], equality_accuracy()));
    }

    /// Adapted from second run example "Robot Localization - A Fully Worked Example" from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
    /// 
    /// Due to the deterministic random number generation used, the loop is far from "blazingly fast".
    /// 
    /// This is almost entirely not the fault of the filter though; it runs much faster without such rng.
    #[test]
    fn unscented_kalman_test_2() {
        assert_eq!(equality_accuracy(), 2);

        let print = false;

        const WHEELBASE : f64 = 0.5;
        const NUM_BEARINGS : usize = 7;

        const STATE_DIM : usize = 3;
        const N_OUT : usize = STATE_DIM * 2 + 1; // aka number of sigma points
        const MEASURE_DIM : usize = NUM_BEARINGS * 2;
        const CONTROL_DIM : usize = 2;
        const TIME_STEP : f64 = 1.; // frequency of running full predict/correct UKF step
        const SIM_STEP : f64 = 0.1; // frequency of updating simulated robot position

        const SIGMA_RANGE : f64 = 0.3;
        const SIGMA_BEARING : f64 = 0.1;
        const MAGIC_NUMBER : f64 = 0.0001; // used in Q

        // Must be of length `NUM_BEARINGS`
        let landmarks = [
            (5., 10.),
            (10., 5.),
            (15., 15.),
            (20., 5.),
            (0., 30.),
            (50., 30.),
            (40., 10.)
        ];

        let init_state_vector =
            SMatrix::<f64, STATE_DIM, 1>::new(2., 6., 0.3);
        let init_estimate_covariance =
            SMatrix::<f64, STATE_DIM, STATE_DIM>::from_diagonal(&Vector3::new(0.1, 0.1, 0.05));
        let measure_covariance =
            SMatrix::<f64, MEASURE_DIM, MEASURE_DIM>::from_diagonal(
                &SMatrix::<f64, MEASURE_DIM, 1>::from_column_slice(
                    &[SIGMA_RANGE.powi(2), SIGMA_BEARING.powi(2)].repeat(NUM_BEARINGS)
                )
            );
        let process_noise_covariance = Some(
            SMatrixTimes::<f64, f64, STATE_DIM, STATE_DIM>::new(SMatrix::identity() * MAGIC_NUMBER, 0)
        );

        let state_transition_function = move |state : SMatrix<f64, STATE_DIM, 1>, time_step : f64, control_o : Option<SMatrix<f64, CONTROL_DIM, 1>>| {
            let curr_angle = state[2];
            
            if let Some(control) = control_o {
                let vel = control[0];
                let steer_angle = control[1];
                let dist = vel * time_step; // euler method

                if steer_angle.abs() > 0.001 {
                    let beta = (dist / WHEELBASE) * steer_angle.tan();
                    let turn_radius = WHEELBASE / steer_angle.tan();

                    let (sinh, sinhb) = (curr_angle.sin(), (curr_angle + beta).sin());
                    let (cosh, coshb) = (curr_angle.cos(), (curr_angle + beta).cos());
                    state + SMatrix::<f64, STATE_DIM, 1>::new(
                        turn_radius * sinhb - turn_radius * sinh,
                        turn_radius * cosh - turn_radius * coshb,
                        beta
                    )
                } else {
                    state + SMatrix::<f64, STATE_DIM, 1>::new(
                        dist * curr_angle.cos(),
                        dist * curr_angle.sin(),
                        0.
                    )
                }
            } else {
                panic!("Impossible") // using control for this example
            }
        };

        // measurement
        let residual_h = |a : SMatrix<f64, MEASURE_DIM, 1>, b : SMatrix<f64, MEASURE_DIM, 1>| {
            let mut residual = a - b;
            for i in (0..residual.len()).step_by(2) {
                residual[i + 1] = normalize_angle(residual[i + 1]);
            }
            residual
        };
        // state
        let residual_x = |a : SMatrix<f64, STATE_DIM, 1>, b : SMatrix<f64, STATE_DIM, 1>| {
            let mut residual = a - b;
            residual[2] = normalize_angle(residual[2]);
            residual
        };

        assert_eq!(landmarks.len(), NUM_BEARINGS);

        // take a state variable and return the measurement that would correspond to that state.
        let observation_function = |state : SMatrix<f64, STATE_DIM, 1>| {
            let mut out = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

            for (i, (px, py)) in landmarks.iter().enumerate() {
                let dist = ((px - state[0]).powi(2) + (py - state[1]).powi(2)).sqrt();
                let angle = (py - state[1]).atan2(px - state[0]);
                out[2 * i] = dist;
                out[2 * i + 1] = normalize_angle(angle - state[2]);
            }

            out
        };

        let state_mean_function = |sigmas : [SMatrix<f64, STATE_DIM, 1>; N_OUT], weights : [f64; N_OUT]| {
            let mut out = SMatrix::<f64, STATE_DIM, 1>::zeros();

            let sum_sin = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[2].sin() * weight)
                .fold(0., |acc, x| acc + x);
            let sum_cos = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[2].cos() * weight)
                .fold(0., |acc, x| acc + x);

            out[0] = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[0] * weight)
                .fold(0., |acc, x| acc + x);
            out[1] = sigmas
                .iter()
                .zip(weights)
                .map(|(sigma, weight)| sigma[1] * weight)
                .fold(0., |acc, x| acc + x);
            out[2] = sum_sin.atan2(sum_cos);

            out
        };

        let measure_mean_function = |sigmas : [SMatrix<f64, MEASURE_DIM, 1>; N_OUT], weights : [f64; N_OUT]| {
            let mut out = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

            for i in (0..MEASURE_DIM).step_by(2) {
                let sum_sin = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i + 1].sin() * weight)
                    .fold(0., |acc, x| acc + x);
                let sum_cos = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i + 1].cos() * weight)
                    .fold(0., |acc, x| acc + x);

                out[i] = sigmas
                    .iter()
                    .zip(weights)
                    .map(|(sigma, weight)| sigma[i] * weight)
                    .fold(0., |acc, x| acc + x);
                out[i + 1] = sum_sin.atan2(sum_cos);
            }

            out
        };

        let add_state_function = |a : SMatrix<f64, STATE_DIM, 1>, b : MatrixView<'_, f64, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>| {
            SMatrix::<f64, STATE_DIM, 1>::new(
                a[0] + b[0],
                a[1] + b[1],
                normalize_angle(a[2] + b[2])
            )
        };

        let mut filter =
            FilterKalmanUnscented::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, _, _, Julier<f64, STATE_DIM, N_OUT>, _, _, _, _, _> {
                state_vector: init_state_vector,
                estimate_covariance: init_estimate_covariance,
                measure_covariance: measure_covariance,
                process_noise_covariance: process_noise_covariance,
                observation: observation_function,
                state_transition: state_transition_function,
                sigma_generator_function: Julier::new(0.),
                state_mean_function: state_mean_function,
                measure_mean_function: measure_mean_function,
                residual_z_function: residual_h,
                residual_x_function: residual_x,
                add_state_function: add_state_function
        };

        let mut commands = vec![];

        // accelerate from a "stop"
        for i in 0..30 {
            let tmp = (i as f64 / 29.) * (1.1 - 0.001) + 0.001;
            commands.push(SMatrix::<f64, CONTROL_DIM, 1>::new(tmp, 0.));
        }
        for _ in 0..50 {
            commands.push(commands[commands.len() - 1]);
        }

        // turn left
        for i in 0..15 {
            let angle = (i as f64 / 14.) * 2. + 0.;
            commands.push(SMatrix::<f64, CONTROL_DIM, 1>::new(1.1, angle.to_radians()));
        }
        for _ in 0..100 {
            commands.push(commands[commands.len() - 1]);
        }

        // turn right
        for i in 0..15 {
            let angle = (i as f64 / 14.) * (-2. - 2.) + 2.;
            commands.push(SMatrix::<f64, CONTROL_DIM, 1>::new(1.1, angle.to_radians()));
        }
        for _ in 0..200 {
            commands.push(commands[commands.len() - 1]);
        }

        for i in 0..15 {
            let angle = (i as f64 / 14.) * (0. - -2.) + -2.;
            commands.push(SMatrix::<f64, CONTROL_DIM, 1>::new(1.1, angle.to_radians()));
        }
        for _ in 0..150 {
            commands.push(commands[commands.len() - 1]);
        }

        for i in 0..25 {
            let angle = (i as f64 / 24.) * (1. - 0.) + 0.;
            commands.push(SMatrix::<f64, CONTROL_DIM, 1>::new(1.1, angle.to_radians()));
        }
        for _ in 0..100 {
            commands.push(commands[commands.len() - 1]);
        }


        let mut sim_pos = init_state_vector;

        let mut rng = ChaCha8Rng::seed_from_u64(76);

        let mut final_estimate = None;

        // The author of the original first robot example did include a `sigma_steer`, but I think they didn't actually use it.
        for (i, command) in commands.iter().enumerate() {
            sim_pos = state_transition_function(sim_pos, SIM_STEP, Some(*command));

            if print { println!("i = {}\tposition:\t{:?}", i, sim_pos); }

            if (i + 1) % (TIME_STEP / SIM_STEP) as usize == 0 {
                let mut measurement_vector = SMatrix::<f64, MEASURE_DIM, 1>::zeros();

                for (j, landmark) in landmarks.iter().enumerate() {
                    let (dx, dy) = (landmark.0 - sim_pos[0], landmark.1 - sim_pos[1]);

                    let randomness : f64 = StandardNormal.sample(&mut rng);
                    let dist = (dx.powi(2) + dy.powi(2)).sqrt() + randomness * SIGMA_RANGE;

                    let randomness : f64 = StandardNormal.sample(&mut rng);
                    let bearing = (landmark.1 - sim_pos[1]).atan2(landmark.0 - sim_pos[0]);
                    let a = normalize_angle(bearing - sim_pos[2] + randomness * SIGMA_BEARING);

                    measurement_vector[2 * j] = dist;
                    measurement_vector[2 * j + 1] = a;
                }

                let processed = filter.process_sample(&KalmanInput::<f64, f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
                    measurement_vector: measurement_vector,
                    control_vector: Some(*command),
                    process_noise_covariance: None,
                    delta_time: Some(TIME_STEP), // not technically necessary since TIME_STEP = 1.
                });

                if print { println!("i = {}\tPROCESSED:\t{:?}", i, processed); }

                if i == commands.len() - 1 {
                    final_estimate = Some(processed);
                }
            }
        }

        assert!(final_estimate.is_some());
        assert_eq!(66.96, round_to_place(final_estimate.unwrap()[0], equality_accuracy()));
        assert_eq!(12.43, round_to_place(final_estimate.unwrap()[1], equality_accuracy()));
        assert_eq!(-0.05, round_to_place(final_estimate.unwrap()[2], equality_accuracy()));
    }
}
