use assert_ok::assert_ok;
use nalgebra_sparse::{CooMatrix};
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;
use shuffle::fy::FisherYates;
use shuffle::shuffler::Shuffler;
use crate::rightfm::Flt;

fn random_train_test_split(
    interactions: &CooMatrix<Flt>,
    test_percentage: f64,
    random_seed: Option<u64>,
) -> (CooMatrix<Flt>, CooMatrix<Flt>) {
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let (uids, iids, data) = (interactions.row_indices(), interactions.col_indices(), interactions.values());

    let mut shuffled_indices = (0..interactions.nnz()).collect::<Vec<_>>();
    assert_ok!(FisherYates::default().shuffle(&mut shuffled_indices, &mut rng));
    let mut _uids = vec![0; interactions.nnz()];
    let mut _iids = vec![0; interactions.nnz()];
    let mut _data = vec![0 as Flt; interactions.nnz()];
    shuffled_indices.iter().for_each(|i| {
        _uids[*i] = uids[*i];
        _iids[*i] = iids[*i];
        _data[*i] = data[*i];
    });
    let cutoff = (uids.len() as f64 * (1.0 - test_percentage)).round() as usize;

    (
        CooMatrix::try_from_triplets(
            interactions.nrows(), interactions.ncols(),
            _uids[..cutoff].to_vec(), _iids[..cutoff].to_vec(), _data[..cutoff].to_vec(),
        ).ok().unwrap(),
        CooMatrix::try_from_triplets(
            interactions.nrows(), interactions.ncols(),
            _uids[cutoff..].to_vec(), _iids[cutoff..].to_vec(), _data[cutoff..].to_vec(),
        ).ok().unwrap()
    )
}