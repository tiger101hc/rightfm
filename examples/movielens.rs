extern crate rightfm;

use std::str::FromStr;

use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rightfm::{LearningSchedule, Loss, RandomState};

fn main() {
    let loss = std::env::args().nth(1).unwrap_or("logistic".to_string());
    let learning_strategy = std::env::args().nth(2).unwrap_or("adagrad".to_string());

    println!(
        "loss strategy is {:?},learning strategy is {:?}",
        loss, learning_strategy
    );
    let data = rightfm::fetch_movielens(None, true, false, 1.0, false);
    let mut model = rightfm::RightFM::new_with_params(
        10,
        5,
        10,
        LearningSchedule::from_str(&learning_strategy).unwrap(),
        Loss::from_str(&loss).unwrap(),
        0.05,
        0.95,
        1e-6,
        0.,
        0.,
        10,
        RandomState::default(),
    );
    model.fit(
        &data.train,
        None,
        Some(data.item_features.clone()),
        None,
        10,
        1,
        true,
    );

    let train_precision = rightfm::precision_at_k(
        &model,
        &CsrMatrix::from(&data.train),
        None,
        10,
        None,
        Some(data.item_features.clone()),
        false,
        1,
        true,
    );
    let test_precision = rightfm::precision_at_k(
        &model,
        &CsrMatrix::from(&data.test),
        Some(CsrMatrix::from(&data.train)),
        10,
        None,
        Some(data.item_features.clone()),
        false,
        1,
        true,
    );
    println!(
        "Precision train:{},test:{}",
        DVector::from_vec(train_precision).mean(),
        DVector::from_vec(test_precision).mean()
    );

    let train_auc = rightfm::auc_score(
        &model,
        &CsrMatrix::from(&data.train),
        None,
        None,
        Some(data.item_features.clone()),
        true,
        1,
        true,
    );
    let test_auc = rightfm::auc_score(
        &model,
        &CsrMatrix::from(&data.test),
        Some(CsrMatrix::from(&data.train)),
        None,
        Some(data.item_features.clone()),
        true,
        1,
        true,
    );
    println!(
        "AUC train:{},test:{}",
        DVector::from_vec(train_auc).mean(),
        DVector::from_vec(test_auc).mean()
    );
}
