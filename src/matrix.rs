use num::traits::Num;
use num::zero;
use std::ops;

#[derive(Clone)]
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

    pub fn map_with<F>(&self, other: Self, func: F) -> Self
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
}

/// Addition implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Add<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.map_with(other, |a, b| a + b)
    }
}

impl<'a, 'b, T: Clone + Num> ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.map_with_by_ref(other, |a, b| a + b)
    }
}

/// Subtraction implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Sub<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.map_with(other, |a, b| a - b)
    }
}

impl<'a, 'b, T: Clone + Num> ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.map_with_by_ref(other, |a, b| a - b)
    }
}

/// Matrix multiplication implementation for Matrix and &Matrix

impl<T: Clone + Num> ops::Mul<Matrix<T>> for Matrix<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::<T>::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut val = result.at(i, j);
                for k in 0..self.cols {
                    val = val + self.at(i, k) * other.at(k, j);
                }
                result.set(i, j, val);
            }
        }

        result
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

/// Tests

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
        let mat = Matrix {
            rows: 2,
            cols: 2,
            data: vec![4, 5, 6, 7],
        };

        let new_mat = mat.map(|val| val * 2);

        assert_eq!(new_mat.at(0, 0), 8);
        assert_eq!(new_mat.at(0, 1), 10);
        assert_eq!(new_mat.at(1, 0), 12);
        assert_eq!(new_mat.at(1, 1), 14);
    }

    #[test]
    fn map_with_test() {
        let mat1 = Matrix::from(2, 2, vec![1, 2, 3, 4]);
        let mat2 = Matrix::from(2, 2, vec![10, 20, 30, 40]);

        let new_mat = mat1.map_with(mat2, |a, b| a + b);

        assert_eq!(new_mat.at(0, 0), 11);
        assert_eq!(new_mat.at(0, 1), 22);
        assert_eq!(new_mat.at(1, 0), 33);
        assert_eq!(new_mat.at(1, 1), 44);
    }

    #[test]
    fn addition_test() {
        let mat1 = Matrix::from(2, 2, vec![10, 20, 30, 40]);
        let mat2 = Matrix::from(2, 2, vec![1, 2, 3, 4]);

        // By reference
        let new_mat = &mat1 + &mat2;

        assert_eq!(new_mat.at(0, 0), 11);
        assert_eq!(new_mat.at(0, 1), 22);
        assert_eq!(new_mat.at(1, 0), 33);
        assert_eq!(new_mat.at(1, 1), 44);

        // By value
        let new_mat = mat1 + mat2;

        assert_eq!(new_mat.at(0, 0), 11);
        assert_eq!(new_mat.at(0, 1), 22);
        assert_eq!(new_mat.at(1, 0), 33);
        assert_eq!(new_mat.at(1, 1), 44);
    }

    #[test]
    fn subtraction_test() {
        let mat1 = Matrix::from(2, 2, vec![11, 22, 33, 44]);
        let mat2 = Matrix::from(2, 2, vec![1, 2, 3, 4]);

        // By reference
        let new_mat = &mat1 - &mat2;

        assert_eq!(new_mat.at(0, 0), 10);
        assert_eq!(new_mat.at(0, 1), 20);
        assert_eq!(new_mat.at(1, 0), 30);
        assert_eq!(new_mat.at(1, 1), 40);

        // By value
        let new_mat = mat1 - mat2;

        assert_eq!(new_mat.at(0, 0), 10);
        assert_eq!(new_mat.at(0, 1), 20);
        assert_eq!(new_mat.at(1, 0), 30);
        assert_eq!(new_mat.at(1, 1), 40);
    }

    #[test]
    fn multiplication_test() {
        let mat1 = Matrix::from(2, 3, vec![3, 4, 5, 1, 6, 8]);
        let mat2 = Matrix::from(3, 2, vec![6, 2, 9, 0, 3, 1]);

        // by reference
        let new_mat = &mat1 * &mat2;

        assert_eq!(new_mat.rows, 2);
        assert_eq!(new_mat.cols, 2);
        assert_eq!(new_mat.at(0, 0), 69);
        assert_eq!(new_mat.at(0, 1), 11);
        assert_eq!(new_mat.at(1, 0), 84);
        assert_eq!(new_mat.at(1, 1), 10);

        // By value
        let new_mat = mat1 * mat2;

        assert_eq!(new_mat.rows, 2);
        assert_eq!(new_mat.cols, 2);
        assert_eq!(new_mat.at(0, 0), 69);
        assert_eq!(new_mat.at(0, 1), 11);
        assert_eq!(new_mat.at(1, 0), 84);
        assert_eq!(new_mat.at(1, 1), 10);
    }
}
