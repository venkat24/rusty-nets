use num::{traits::Num, zero};
use std::ops;

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone + Num> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows: rows,
            cols: cols,
            data: vec![zero(); rows * cols],
        }
    }

    pub fn from(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Self {
            rows: rows,
            cols: cols,
            data: data,
        }
    }

    pub fn index(&self, i: usize, j: usize) -> usize {
        (i * self.cols) + j
    }

    pub fn at(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows);
        assert!(j < self.cols);

        self.data[self.index(i, j)].clone()
    }

    pub fn set(&mut self, i: usize, j: usize, val: T) {
        assert!(i < self.rows);
        assert!(j < self.cols);

        let index = self.index(i, j);
        self.data[index] = val;
    }

    pub fn map<F>(&self, func: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let mut result = Matrix::<T>::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = func(self.at(i, j));
                result.set(i, j, val);
            }
        }

        result
    }

    pub fn map_with_by_ref<F>(&self, other: &Matrix<T>, func: F) -> Matrix<T>
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::<T>::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = func(self.at(i, j), other.at(i, j));
                result.set(i, j, val);
            }
        }

        result
    }

    pub fn map_with<F>(&self, other: Self, func: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        self.map_with_by_ref(&other, func)
    }
}

// Equality comparisons for Matrix

impl<T: Clone + Num> PartialEq for Matrix<T> {
    fn eq(&self, other: &Matrix<T>) -> bool {
        // Directly compare the slices of the two vectors
        &self.data[..] == &other.data[..]
    }
}

// Addition implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Add<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        &self + &other
    }
}

impl<'a, 'b, T: Clone + Num> ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.map_with_by_ref(other, |a, b| a + b)
    }
}

// Subtraction implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Sub<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        &self - &other
    }
}

impl<'a, 'b, T: Clone + Num> ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.map_with_by_ref(other, |a, b| a - b)
    }
}

// Matrix multiplication implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Mul<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        &self * &other
    }
}

impl<'a, 'b, T: Clone + Num> ops::Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::<T>::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut val = num::zero();
                for k in 0..self.cols {
                    val = val + self.at(i, k) * other.at(k, j);
                }
                result.set(i, j, val);
            }
        }

        result
    }
}

// Macros

#[macro_export]
macro_rules! sq_matrix {
    ($elem:expr; $n:expr) => {
        {
            let size = $n as usize;
            let data = vec![$elem; size * size];

            Matrix {
                rows: size,
                cols: size,
                data: data
            }
        }
    };

    ( $( $x:expr ),* ) => {
        {
            use crate::utils::get_integral_square_root;

            let data_vec = vec![$($x),*];
            let data_len = data_vec.len();

            // Ensure that number of elements is a perfect square
            match get_integral_square_root(data_len) {
                Some(root) => Matrix {
                    rows: root,
                    cols: root,
                    data: data_vec
                },
                None => panic!("Number of elements must be a perfect square..")
            }
        }
    };
}

// Tests

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn init_test() {
        let m = 10;
        let n = 20;
        let mat = Matrix::<i64>::new(m, n);

        for i in 0..m {
            for j in 0..n {
                assert_eq!(mat.at(i, j), 0);
            }
        }
    }

    #[test]
    fn setter_test() {
        let mut mat = Matrix::<i64>::new(2, 2);

        mat.set(0, 0, 4);
        mat.set(0, 1, 5);
        mat.set(1, 0, 6);
        mat.set(1, 1, 7);

        assert_eq!(mat.at(0, 0), 4);
        assert_eq!(mat.at(0, 1), 5);
        assert_eq!(mat.at(1, 0), 6);
        assert_eq!(mat.at(1, 1), 7);
    }

    #[test]
    fn map_test() {
        let mat = sq_matrix![4, 5, 6, 7];
        let expected = sq_matrix![8, 10, 12, 14];

        let new_mat = mat.map(|val| val * 2);
        assert_eq!(new_mat, expected);
    }

    #[test]
    fn map_with_test() {
        let mat1 = sq_matrix![1, 2, 3, 4];
        let mat2 = sq_matrix![10, 20, 30, 40];

        let expected = sq_matrix![11, 22, 33, 44];

        let new_mat = mat1.map_with(mat2, |a, b| a + b);
        assert_eq!(new_mat, expected);
    }

    #[test]
    fn addition_test() {
        let mat1 = sq_matrix![10, 20, 30, 40];
        let mat2 = sq_matrix![1, 2, 3, 4];

        let expected = sq_matrix![11, 22, 33, 44];

        // By reference
        let new_mat = &mat1 + &mat2;
        assert_eq!(new_mat, expected);

        // By value
        let new_mat = mat1 + mat2;
        assert_eq!(new_mat, expected);
    }

    #[test]
    fn subtraction_test() {
        let mat1 = sq_matrix![11, 22, 33, 44];
        let mat2 = sq_matrix![1, 2, 3, 4];

        let expected = sq_matrix![10, 20, 30, 40];

        // By reference
        let new_mat = &mat1 - &mat2;
        assert_eq!(new_mat, expected);

        // By value
        let new_mat = mat1 - mat2;
        assert_eq!(new_mat, expected);
    }

    #[test]
    fn multiplication_test() {
        let mat1 = Matrix::from(2, 3, vec![3, 4, 5, 1, 6, 8]);
        let mat2 = Matrix::from(3, 2, vec![6, 2, 9, 0, 3, 1]);

        let expected = sq_matrix![69, 11, 84, 10];

        // By reference
        let new_mat = &mat1 * &mat2;
        assert_eq!(new_mat, expected);

        // By value
        let new_mat = mat1 * mat2;
        assert_eq!(new_mat, expected);
    }
}
