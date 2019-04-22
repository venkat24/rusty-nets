pub fn get_integral_square_root(val: usize) -> Option<usize> {
    let square_root = (val as f64).sqrt();

    if square_root.fract() != 0. {
        return None;
    }

    Some(square_root as usize)
}
