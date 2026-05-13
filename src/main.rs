use crate::signal::Signal;

mod consts;
mod signal;



fn main() {
    let s1 = Signal::from(
        [1, 0, 0, 0].as_slice()
    );
    let s3 = s1.clone();
    let s4 = s1.forward_dft();

    println!("signal: {}\ndft: {}", s3, s4);
}

#[cfg(test)]
mod tests {
    use crate::signal::Signal;

    #[test]
    fn linearity() {
        let cos = Signal::from([1, 0, -1, 0].as_slice());
        let dc = Signal::from([1, 1, 1, 1].as_slice());

        let mut cos_dft = cos.forward_dft();
        let dc_dft = dc.forward_dft();

        

        let cos_plus_dc = Signal::from([2, 1, 0, 1].as_slice());

        let combine_dft = cos_plus_dc.forward_dft();

        println!("combine_dft {}", combine_dft);
        println!("cos_dft + &dc_dft {}", &mut cos_dft + &dc_dft);

        assert!(combine_dft == cos_dft);
    }
}