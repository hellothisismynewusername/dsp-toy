use std::{f64::consts::PI, fmt::Display, rc::Rc};

use easy_complex::{Complex, Complex64};

const EULER : f64 = 2.718281828459045235360287471352;

#[derive(Debug)]
struct Signal {
    pub data : Rc<[Complex64]>,
    approx : bool,
}

impl Signal {
    pub fn new() -> Signal {
        Signal {
            data: Rc::new([]),
            approx: false
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn forward_dft(&self) -> Signal {
        let mut data_tmp : Vec<Complex64> = Vec::new();
        for k in 0..self.len() {
            // polar form but outputted signal will be in cartesian
            let tmp = self.data.iter().enumerate().map(|(n, val)| {
                *val * Complex64::from(EULER).powc(
                        Complex64::from(-1.) * j() * (Complex64::from(2. * PI) / Complex64::from(self.len() as f64)) * Complex64::from(k as f64) * Complex64::from(n as f64)
                    )
            }).reduce(|sum, x| {
                sum + x
            });

            data_tmp.push(tmp.unwrap());
        }

        let mut out = Signal::from(data_tmp.as_slice());
        if self.approx {
            out.approx = true;
        }
        out
    }

    pub fn round(&self) -> Signal {
        let tmp : Vec<Complex64> = self.data.iter().map(|val| {
            Complex64::new(round_to_place(val.real(), 3), round_to_place(val.imag(), 3))
        }).collect();
        Signal {
            data: Rc::from(tmp.as_slice()),
            approx: true
        }
    }
}

impl From<&[Complex64]> for Signal {
    fn from(data: &[Complex64]) -> Self {
        Signal { data: Rc::from(data), approx: false }
    }
}

impl Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = "[".to_string();
        for (i, val) in self.data.iter().enumerate() {
            if i != 0 {
                out += ", ";
            }
            let mut mag = f64::sqrt(val.real().powf(2.) + val.imag().powf(2.));
            let mut phase = f64::atan2(val.imag(), val.real());
            if self.approx {
                mag = round_to_place(mag, 3);
                phase = round_to_place(phase, 3);
            }
            out += &*("".to_string() + &*mag.to_string() + " * e^" + &*phase.to_string());
        }
        out += "]";
        
        write!(f, "{}", out)
    }
}

fn main() {
    let s1 = Signal::from(
        [1.into(), 0.into(), (-1).into(), 0.into()].as_slice()
    );
    println!("signal: {}\ndft: {}", s1.round(), s1.forward_dft().round());
}

fn j() -> Complex64 {
    Complex64::new(0., 1.)
}

fn round_to_place(num : f64, place : i32) -> f64 {
    let factor = 10_f64.powi(place);
    (num * factor).round() / factor
}