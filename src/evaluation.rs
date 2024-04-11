use nalgebra_sparse::CsrMatrix;
use ndarray::Array1;
use crate::{RightFM, rightfm_fast::{calculate_auc_from_rank}};
use crate::rightfm::Flt;

pub fn precision_at_k(
    model: &RightFM,
    test_interactions: &CsrMatrix<Flt>,
    train_interactions: Option<CsrMatrix<Flt>>,
    k: usize,
    user_features: Option<CsrMatrix<Flt>>,
    item_features: Option<CsrMatrix<Flt>>,
    preserve_rows: bool,
    num_threads: usize,
    check_intersections: bool,
) -> Vec<Flt> {
    assert!(num_threads > 0, "num_threads must be greater than 0");
    let ranks = model.predict_rank(test_interactions, train_interactions, user_features, item_features, num_threads, check_intersections);
    let ranks = CsrMatrix::try_from_csr_data(ranks.nrows(),
                                             ranks.ncols(), ranks.row_offsets().to_vec(),
                                             ranks.col_indices().to_vec(),
                                             ranks.values().iter().map(|v| if v >= &(k as Flt) { 0 } else { 1 }).collect::<Vec<usize>>()).unwrap();
    let precision = (0..ranks.nrows()).map(|i| {
        if ranks.get_row(i).is_none() { 0 as Flt } else {
            (ranks.get_row(i).unwrap().values().iter().sum::<usize>() / k) as Flt
        }
    }).collect::<Vec<Flt>>();
    if preserve_rows {
        precision
    } else {
        precision.iter().enumerate().filter(|(i, _)| ranks.get_row(*i).is_some()).map(|(_, v)| *v).collect()
    }
}

pub fn recall_at_k(
    model: &RightFM,
    test_interactions: &CsrMatrix<Flt>,
    train_interactions: Option<CsrMatrix<Flt>>,
    k: usize,
    user_features: Option<CsrMatrix<Flt>>,
    item_features: Option<CsrMatrix<Flt>>,
    preserve_rows: bool,
    num_threads: usize,
    check_intersections: bool,
) -> Vec<Flt> {
    assert!(num_threads > 0, "num_threads must be greater than 0");
    let ranks = model.predict_rank(test_interactions, train_interactions, user_features, item_features, num_threads, check_intersections);
    let ranks = CsrMatrix::try_from_csr_data(ranks.nrows(),
                                             ranks.ncols(), ranks.row_offsets().to_vec(),
                                             ranks.col_indices().to_vec(),
                                             (0..ranks.nnz()).map(
                                                 |i| if ranks.values()[i] >= k as Flt { 0 } else { 1 }
                                             ).collect::<Vec<usize>>()).unwrap();
    let recall = (0..ranks.nrows()).map(|i| {
        if ranks.get_row(i).is_none() { 0 as Flt } else {
            (ranks.get_row(i).unwrap().values().iter().sum::<usize>() / ranks.get_row(i).unwrap().nnz()) as Flt
        }
    }).collect::<Vec<Flt>>();
    if preserve_rows { recall } else {
        recall.iter().enumerate().filter(|(i, _)| ranks.get_row(*i).is_some()).map(|(_, v)| *v).collect()
    }
}

pub fn auc_score(
    model: &RightFM,
    test_interactions: &CsrMatrix<Flt>,
    train_interactions: Option<CsrMatrix<Flt>>,
    user_features: Option<CsrMatrix<Flt>>,
    item_features: Option<CsrMatrix<Flt>>,
    preserve_rows: bool,
    num_threads: usize,
    check_intersections: bool,
) -> Vec<Flt> {
    assert!(num_threads > 0, "num_threads must be greater than 0");
    let num_train_positives = if train_interactions.is_none() { vec![0; test_interactions.nrows()] } else {
        let _ti = train_interactions.as_ref().unwrap();
        (0.._ti.nrows()).map(|i| if _ti.get_row(i).is_none() {0} else {_ti.get_row(i).as_ref().unwrap().nnz()}).collect()
    };
    let _train_interactions = if train_interactions.is_none() { Some(CsrMatrix::zeros(test_interactions.nrows(), test_interactions.ncols())) } else { train_interactions };
    let ranks = model.predict_rank(test_interactions, _train_interactions, user_features, item_features, num_threads, check_intersections);
    let mut auc = vec![0 as Flt; ranks.nrows()];
    calculate_auc_from_rank(&ranks, &num_train_positives, &mut ranks.values().to_vec(), &mut auc, num_threads);
    if preserve_rows { auc } else {
        auc.iter().enumerate().filter(|(i, _)| test_interactions.get_row(*i).is_some()).map(|(_, v)| *v).collect::<Vec<Flt>>()
    }
}

pub fn reciprocal_rank(
    model: &RightFM,
    test_interactions: &CsrMatrix<Flt>,
    train_interactions: Option<CsrMatrix<Flt>>,
    user_features: Option<CsrMatrix<Flt>>,
    item_features: Option<CsrMatrix<Flt>>,
    preserve_rows: bool,
    num_threads: usize,
    check_intersections: bool,
) -> Vec<Flt> {
    assert!(num_threads > 0, "num_threads must be greater than 0");
    let ranks = model.predict_rank(test_interactions, train_interactions, user_features, item_features, num_threads, check_intersections);
    let data = Array1::from_vec(ranks.values().to_vec());
    let ranks: CsrMatrix<Flt> = CsrMatrix::try_from_csr_data(ranks.nrows(), ranks.ncols(),
                                                             ranks.row_offsets().to_vec(), ranks.col_indices().to_vec(), (Array1::ones(ranks.nnz()) / (data + 1 as Flt)).to_vec()).unwrap();
    let rank = (0..ranks.nrows()).map(|i|
        if ranks.get_row(i).is_none() { 0 as Flt } else {
            let mut max = 0 as Flt;
            ranks.get_row(i).unwrap().values().iter().for_each(|v| {
                max = v.max(max.clone());
            });
            max
        }
    ).collect::<Vec<Flt>>();
    if preserve_rows {
        rank
    } else {
        rank.iter().enumerate().filter(|(i, _)| ranks.get_row(*i).is_some()).map(|(_, v)| v.clone()).collect::<Vec<Flt>>()
    }
}