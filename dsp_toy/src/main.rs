use dsp_toy_lib::signal::Signal;

fn main() {
    let s1 = Signal::from([1, 2]);
    let s2 = Signal::from([3, 4]);

    println!("{}", s1.convolve(&s2));

}