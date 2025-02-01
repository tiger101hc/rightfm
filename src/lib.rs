mod cross_validation;
mod datasets;
mod evaluation;
mod rightfm;
mod rightfm_fast;

pub use cross_validation::*;
pub use datasets::*;
pub use evaluation::*;
pub use rightfm::*;

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, MatrixXx2};
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    #[test]
    fn test1() {
        let myarr2: Array2<f32> = Array2::random((3, 3), Uniform::new(0.0, 1.0));
        let r0 = myarr2.row(0);
        println!("row 0 is {:?}", r0);
        let matrix = DMatrix::from_row_slice(3, 3, &myarr2.as_slice().unwrap());
        let matrix2 = MatrixXx2::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        println!("{}\r\n{}", matrix, matrix2)
    }

    #[test]
    fn test2() {
        println!("hello test222222");
        // let dm = Matrix2::<f64>::identity();
        // let dm2 = (&dm - 1.) / 2.;
        // println!("{}/r/n{}", dm, dm2);
        let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        println!("Matrix:");
        println!("{}", matrix);

        // 标量加法
        let scalar_add = matrix.clone() * 2.0;
        println!("Matrix + 2.0:");
        println!("{}", scalar_add);

        let uniform = Uniform::new(-0.5 as f32, 0.5 as f32);
        let mm = DMatrix::from_distribution(2, 10, &uniform, &mut rand::thread_rng());
        println!("{}", mm)
    }

    #[test]
    fn test3() {
        let a1: Array1<f32> = Array1::from_vec(vec![1., 2., 3.]);
        let a2: Array1<f32> = Array1::ones(3);
        let r = a2.clone() / (a1.clone() + 1.);
        println!("{},{},{}", a1, a2, r);

        let n1 = DVector::<f32>::from_vec(vec![1., 2., 3.]);
        let n2 = DVector::<f32>::repeat(3, 1.);
        let n3 = n1.clone() + n2.clone();
        let rr = n2.component_div(&n3);
        println!("{}", rr);
    }
}
