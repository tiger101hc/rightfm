extern crate rightfm;

use nalgebra_sparse::CsrMatrix;
use ndarray::Array1;
use rightfm::{LearningSchedule, Loss, RandomState};

fn main() {
    let data = rightfm::fetch_movielens(None, true, false, 1.0, false);
    let mut model = rightfm::RightFM::new_with_params(10,5,10,LearningSchedule::AdaDelta,Loss::Warp,0.05,0.95,
    1e-6,0.,0.,10,RandomState::default());
    model.fit(&data.train, None, Some(data.item_features.clone()), None, 10, 1, true);

    let train_precision = rightfm::precision_at_k(&model, &CsrMatrix::from(&data.train),
                                                  None, 10, None,
                                                  Some(data.item_features.clone()),
                                                  false, 1, true);
    let test_precision = rightfm::precision_at_k(&model, &CsrMatrix::from(&data.test),
                                                 Some(CsrMatrix::from(&data.train)),
                                                 10, None, Some(data.item_features.clone()),
                                                 false, 1, true);
    println!("Precision train:{},test:{}", Array1::from_vec(train_precision).mean().unwrap(), Array1::from_vec(test_precision).mean().unwrap());

    let train_auc = rightfm::auc_score(&model, &CsrMatrix::from(&data.train),
                                       None, None, Some(data.item_features.clone()),
                                       true, 1, true);
    let test_auc = rightfm::auc_score(&model, &CsrMatrix::from(&data.test),
                                      Some(CsrMatrix::from(&data.train)), None,
                                      Some(data.item_features.clone()),
                                      true, 1, true);
    println!("AUC train:{},test:{}", Array1::from_vec(train_auc).mean().unwrap(), Array1::from_vec(test_auc).mean().unwrap());
}
