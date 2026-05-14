use std::ops::Mul;

use crate::signal::Signal;

mod consts;
mod signal;



fn main() {
    let signal = Signal::from(
        [1, 0, -1, 0].as_slice()
    ).zero_extend_end(10);
    let ir = Signal::from(
        [1., 0.2, 0.15, 0.].as_slice()
    ).zero_extend_end(10);
    
    println!("signal: {}, ir: {}", signal, ir);

    let ir_dft = ir.forward_dft();
    let signal_final = signal.forward_dft().mul(&ir_dft).inverse_dft();

    println!("signal final {}", signal_final);
}

#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use crate::signal::Signal;

    #[test]
    fn convolve_with_impulse_is_unchanged() {
        let signal = Signal::from(
            [1, 0, -1, 0].as_slice()
        );
        let ir = Signal::from(
            [1, 0, 0, 0].as_slice()
        );
        
        println!("signal: {}, ir: {}", signal, ir);

        let ir_dft = ir.forward_dft();
        let signal_final = signal.forward_dft().mul(&ir_dft).inverse_dft();

        println!("signal final {}", signal_final);

        assert_eq!(signal, signal_final)
    }

    #[test]
    fn inverse() {
        let s = Signal::from([0, 1, 0, -1].as_slice());
        println!("s:\t\t{}\ns_dft:\t\t{}\ns_dft_idft:\t{}", s, s.forward_dft(), s.forward_dft().inverse_dft());
        assert_eq!(s, s.forward_dft().inverse_dft());
    }

    #[test]
    fn linearity() {
        let signal = Signal::from([1, 0, -1, 0].as_slice());
        let dc = Signal::from([1, 1, 1, 1].as_slice());

        let mut signal_dft = signal.forward_dft();
        let dc_dft = dc.forward_dft();

        let signal_plus_dc = Signal::from([2, 1, 0, 1].as_slice());

        let combine_dft = signal_plus_dc.forward_dft();

        println!("combine_dft {}", combine_dft);
        println!("cos_dft + &dc_dft {}", &mut signal_dft + &dc_dft);

        assert!(combine_dft == signal_dft);
    }
}