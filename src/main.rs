fn main() {
    println!("Hello, world!");

    let mut v: Vec<Flt> = vec![1., 2., 3., 4., 5.];
    // v[3] = 9.0;
    println!("{:?}", &v);

    partial_sort(&mut v, 1, 4, true);

    println!("{:?}", &v);
    // let no_item_features = 10;
    // let no_components = 5;
    // let data: ArrayBase<OwnedRepr<Flt>, Dim<[usize; 2]>> = Array2::from_elem((no_item_features, no_components), 0.5 as Flt);
    // println!("{}", data);
    //
    // let mut coo = CooMatrix::<f64>::new(3, 3);
    // coo.push(0, 2, 1.0);
    // coo.push(0, 1, 1.0);
    // coo.push(2, 0, 1.0);
    // println!("{:?}", coo);
    //
    // let csr = CsrMatrix::from(&coo);
    // println!("{:?},{:?},{:?}", csr, csr.col_indices(), csr.get_row(0).unwrap().col_indices());
}

use std::{cmp};
use assert_ok::assert_ok;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use nalgebra_sparse::csr::CsrRow;
use ndarray::{Array, Array1, Array2, ArrayBase, Dim, OwnedRepr, s};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::{Rng, RngCore};
use rand::prelude::ThreadRng;
use rand::rngs::mock::StepRng;
use rayon::iter::IntoParallelIterator;
use shuffle::fy::FisherYates;
use shuffle::shuffler::Shuffler;
use rayon::iter::ParallelIterator;
use tqdm::Iter;


type Flt = f32;

static MAX_REG_SCALE: Flt = 1000000.0;
static MAX_LOSS: Flt = 10.0;


#[derive(Eq, PartialEq, Clone)]
enum LearningSchedule {
    AdaGrad,
    AdaDelta,
}

#[derive(Eq, PartialEq, Clone)]
enum Loss {
    Logistic,
    Warp,
    Bpr,
    WarpKos,
}

pub struct RightFM {
    no_components: usize,
    k: i32,
    n: i32,
    learning_schedule: LearningSchedule,
    loss: Loss,
    learning_rate: Flt,
    rho: Flt,
    epsilon: f64,
    item_alpha: Flt,
    user_alpha: Flt,
    max_sampled: usize,
    random_state: RandomState,
    data: Option<FastRightFM>,
}

pub struct RandomState {
    step_rng: StepRng,
    fy: FisherYates,
    thread_rng: ThreadRng,
}

impl RightFM {
    pub fn new() -> RightFM {
        RightFM {
            no_components: 10,
            k: 5,
            n: 10,
            learning_schedule: LearningSchedule::AdaGrad,
            loss: Loss::Logistic,
            learning_rate: 0.05,
            rho: 0.95,
            epsilon: 1e-6,
            item_alpha: 0.0,
            user_alpha: 0.0,
            max_sampled: 10,
            random_state: RandomState {
                step_rng: StepRng::new(2, 13),
                fy: FisherYates::default(),
                thread_rng: rand::thread_rng(),
            },
            data: None,
        }
    }

    pub fn new_with_params(
        no_components: usize,
        k: i32,
        n: i32,
        learning_schedule: LearningSchedule,
        loss: Loss,
        learning_rate: Flt,
        rho: Flt,
        epsilon: f64,
        item_alpha: Flt,
        user_alpha: Flt,
        max_sampled: usize,
        random_state: RandomState,
    ) -> RightFM {
        assert!(item_alpha >= 0.0);
        assert!(user_alpha >= 0.0);
        assert!(no_components > 0);
        assert!(k > 0);
        assert!(n > 0);
        assert!((0 as Flt) < rho && rho < (1 as Flt));
        assert!(epsilon >= 0f64);

        RightFM {
            no_components,
            k,
            n,
            learning_schedule,
            loss,
            learning_rate,
            rho,
            epsilon,
            item_alpha,
            user_alpha,
            max_sampled,
            random_state,
            data: None,
        }
    }

    pub fn reset_state(&mut self) {
        self.data = None;
    }

    pub fn fit(&mut self, interaction: &CooMatrix<Flt>,
               user_features: &CsrMatrix<Flt>, item_features: &CsrMatrix<Flt>,
               sample_weight: Option<CooMatrix<Flt>>, epochs: usize, number_thread: u8, verbose: bool) {
        assert!(number_thread >= 1);
        assert!(interaction.nrows() > user_features.nrows(), "interaction matrix must have more rows than user features matrix");
        assert!(interaction.ncols() > item_features.nrows(), "interaction matrix must have more cols than item features matrix");
        self.reset_state();
        self.fit_partial(interaction, user_features, item_features, sample_weight, epochs, number_thread, verbose);
    }

    pub fn fit_partial(&mut self, interaction: &CooMatrix<Flt>, user_features: &CsrMatrix<Flt>, item_features: &CsrMatrix<Flt>,
                       sample_weight: Option<CooMatrix<Flt>>, epochs: usize, number_thread: u8, verbose: bool) {
        let sample_weight_data = self.process_sample_weight(interaction, sample_weight);
        if (self.data.is_none()) {
            self.data = Some(
                FastRightFM::new(self.no_components, user_features.ncols(), item_features.ncols(), self.learning_schedule.clone(), self.learning_rate, self.rho, self.epsilon, self.max_sampled)
            );
        } else {
            assert_eq!(self.data.as_ref().unwrap().item_features.nrows(), item_features.ncols(), "should have same no of item features");
            assert_eq!(self.data.as_ref().unwrap().user_features.nrows(), user_features.ncols(), "should have same no of user features");
        }

        Self::process(epochs, verbose).for_each(|epoch| {
            self.run_epoch(item_features, user_features, interaction, &sample_weight_data, epoch, number_thread, self.loss.clone())
        });
    }

    pub fn run_epoch(&mut self, item_features: &CsrMatrix<Flt>, user_features: &CsrMatrix<Flt>, interaction: &CooMatrix<Flt>, sample_weight: &Vec<Flt>, epoch: usize, number_thread: u8, loss: Loss) {
        println!("Epoch {}", epoch);
        let mut shuffled_indices = (0..interaction.values().len()).collect::<Vec<_>>();
        assert_ok!(self.random_state.fy.shuffle(&mut shuffled_indices, &mut self.random_state.step_rng));

        match loss {
            Loss::Logistic => {
                fit_logistic(item_features,
                             user_features,
                             interaction.row_indices(),
                             interaction.col_indices(),
                             interaction.values(),
                             sample_weight,
                             shuffled_indices.as_slice(),
                             self.data.as_mut().unwrap(),
                             self.learning_rate, self.item_alpha, self.user_alpha, number_thread)
            }
            Loss::Warp => {
                fit_warp(item_features,
                         user_features,
                         &CsrMatrix::from(interaction),
                         interaction.row_indices(),
                         interaction.col_indices(),
                         interaction.values(),
                         sample_weight,
                         shuffled_indices.as_slice(),
                         self.data.as_mut().unwrap(),
                         self.learning_rate, self.item_alpha, self.user_alpha, &mut self.random_state);
            }
            Loss::Bpr => {
                fit_bpr(item_features,
                        user_features,
                        &CsrMatrix::from(interaction),
                        interaction.row_indices(),
                        interaction.col_indices(),
                        interaction.values(),
                        sample_weight,
                        shuffled_indices.as_slice(),
                        self.data.as_mut().unwrap(),
                        self.learning_rate, self.item_alpha, self.user_alpha, &mut self.random_state);
            }
            Loss::WarpKos => {
                fit_warp_kos(item_features,
                             user_features,
                             &CsrMatrix::from(interaction),
                             interaction.row_indices(),
                             shuffled_indices.as_slice(),
                             self.data.as_mut().unwrap(),
                             self.learning_rate, self.item_alpha, self.user_alpha, self.k as usize, self.n as usize, &mut self.random_state);
            }
        }
    }

    fn process(n: usize, verbose: bool) -> Box<dyn Iterator<Item=usize>> {
        if verbose { Box::new((0..n).tqdm()) } else { Box::new((0..n)) }
    }

    pub fn process_sample_weight(&self, interaction: &CooMatrix<Flt>, sample_weight: Option<CooMatrix<Flt>>) -> Vec<Flt> {
        if sample_weight.is_some() {
            let data = sample_weight.clone().unwrap();
            assert!(self.loss != Loss::WarpKos, "k-OS loss with sample weights not implemented.");
            assert!(data.nrows() == interaction.nrows() && data.ncols() == interaction.ncols(), "sample weight and interactions matrices must be the same shape");
            return data.values().to_vec();
        }
        return Array::ones(interaction.nnz()).to_vec();
    }

    fn _predict(&self,user_ids: &[usize],item_ids: &[usize], user_features: &CsrMatrix<Flt>, item_features: &CsrMatrix<Flt>) -> Array2<Flt> {
        assert_eq!(user_ids.len(), item_ids.len(), "user_ids and item_ids must be the same length");
        assert!(*user_ids.iter().min().unwrap()>0usize && *item_ids.iter().min().unwrap()>0usize, "user_ids and item_ids must be positive");
        let no_users = user_ids.iter().max().unwrap()+1;
        let no_items = item_ids.iter().max().unwrap()+1;
        assert_eq!(no_users, user_features.nrows(), "user_ids and user_features must have the same no of users");
        assert_eq!(no_items, item_features.nrows(), "item_ids and item_features must have the same no of items");

        let mut predictions = Array2::zeros((user_ids.len(),item_ids.len()));
        predict_rightfm(user_features,item_features,user_ids,item_ids,&mut predictions,self.data.as_ref().unwrap(),1);
        predictions
    }
}

pub struct FastRightFM {
    item_features: Array2<Flt>,
    item_features_gradients: Array2<Flt>,
    item_features_momentum: Array2<Flt>,
    item_biases: Array1<Flt>,
    item_biases_gradients: Array1<Flt>,
    item_biases_momentum: Array1<Flt>,

    user_features: Array2<Flt>,
    user_features_gradients: Array2<Flt>,
    user_features_momentum: Array2<Flt>,
    user_biases: Array1<Flt>,
    user_biases_gradients: Array1<Flt>,
    user_biases_momentum: Array1<Flt>,

    no_components: usize,
    learning_schedule: LearningSchedule,
    learning_rate: Flt,
    rho: Flt,
    eps: f64,
    max_sampled: usize,
    item_scale: Flt,
    user_scale: Flt,
}

impl FastRightFM {
    pub fn new(
        no_components: usize,
        no_item_features: usize,
        no_user_features: usize,
        learn_schedule: LearningSchedule,
        learning_rate: Flt,
        rho: Flt,
        eps: f64,
        max_sampled: usize) -> FastRightFM {
        FastRightFM {
            item_features: (Array2::random((no_item_features, no_components), Uniform::new(0.0 as Flt, 1.0 as Flt)) - 0.5 as Flt) / no_components as Flt,
            item_features_gradients: match learn_schedule {
                LearningSchedule::AdaGrad => {
                    Array2::zeros((no_item_features, no_components)) + 1 as Flt
                }
                LearningSchedule::AdaDelta => {
                    Array2::zeros((no_item_features, no_components))
                }
            },
            item_features_momentum: Array2::zeros((no_item_features, no_components)),
            item_biases: Array1::zeros(no_item_features),
            item_biases_gradients: match learn_schedule {
                LearningSchedule::AdaGrad => { Array1::zeros(no_item_features) + 1. }
                LearningSchedule::AdaDelta => { Array1::zeros(no_item_features) }
            },
            item_biases_momentum: Array1::zeros(no_item_features),
            user_features: (Array2::random((no_user_features, no_components), Uniform::new(0.0 as Flt, 1.0 as Flt)) - (0.5 as Flt)) / no_components as Flt,
            user_features_gradients: match learn_schedule {
                LearningSchedule::AdaGrad => { Array2::zeros((no_user_features, no_components)) + 1. }
                LearningSchedule::AdaDelta => { Array2::zeros((no_user_features, no_components)) }
            },
            user_features_momentum: Array2::zeros((no_user_features, no_components)),
            user_biases: Array1::zeros(no_user_features),
            user_biases_gradients: match learn_schedule {
                LearningSchedule::AdaGrad => { Array1::zeros(no_user_features) + 1. }
                LearningSchedule::AdaDelta => { Array1::zeros(no_user_features) }
            },
            user_biases_momentum: Array1::zeros(no_user_features),
            no_components,
            learning_schedule: learn_schedule,
            learning_rate,
            rho,
            eps,
            max_sampled,
            item_scale: 1.0,
            user_scale: 1.0,
        }
    }
}

fn fit_logistic(
    item_features: &CsrMatrix<Flt>,
    user_features: &CsrMatrix<Flt>,
    user_ids: &[usize],
    item_ids: &[usize],
    values: &[Flt],
    sample_weight: &Vec<Flt>,
    shuffle_indices: &[usize],
    rightfm: &mut FastRightFM,
    learning_rate: Flt,
    item_alpha: Flt,
    user_alpha: Flt,
    num_threads: u8,
) {
    let no_examples = values.len();
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut it_repr = vec![0.0; rightfm.no_components + 1];

    (0..no_examples).for_each(|i| {
        let row = shuffle_indices[i as usize];
        let user_id = user_ids[row];
        let item_id = item_ids[row];
        let weight = sample_weight[row];
        let item_feature = item_features.get_row(item_id).unwrap();
        let user_feature = user_features.get_row(user_id).unwrap();

        compute_representation(&item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut it_repr);
        compute_representation(&user_feature, &rightfm.user_features, &rightfm.user_biases, rightfm, rightfm.user_scale, &mut user_repr);
        let prediction = sigmoid(compute_prediction_from_repr(&user_repr, &it_repr, rightfm.no_components));

        let v_row = values[row];
        let v = if v_row <= 0.0 { 0 } else { 1 };
        let loss = weight * (prediction - v as Flt);

        update(loss, &item_feature, &user_feature, &user_repr, &it_repr, rightfm, item_alpha, user_alpha);
        locked_regularize(rightfm, item_alpha, user_alpha);
    });
    regularize(rightfm, item_alpha, user_alpha);
}

fn fit_warp(
    item_features: &CsrMatrix<Flt>,
    user_features: &CsrMatrix<Flt>,
    interactions: &CsrMatrix<Flt>,
    user_ids: &[usize],
    item_ids: &[usize],
    values: &[Flt],
    sample_weight: &Vec<Flt>,
    shuffle_indices: &[usize],
    rightfm: &mut FastRightFM,
    learning_rate: Flt,
    item_alpha: Flt,
    user_alpha: Flt,
    random_state: &mut RandomState) {
    let no_examples = values.len();
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut pos_it_repr = vec![0.0; rightfm.no_components + 1];
    let mut neg_it_repr = vec![0.0; rightfm.no_components + 1];

    (0..no_examples).for_each(|i| {
        let row = shuffle_indices[i];
        if values[row] > 0.0 {
            let user_id = user_ids[row];
            let positive_item_id = item_ids[row];
            let weight = sample_weight[row];
            let pos_item_feature = item_features.get_row(positive_item_id).unwrap();
            let user_feature = user_features.get_row(user_id).unwrap();

            compute_representation(&pos_item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut pos_it_repr);
            compute_representation(&user_feature, &rightfm.user_features, &rightfm.user_biases, rightfm, rightfm.user_scale, &mut user_repr);
            let pos_prediction = compute_prediction_from_repr(&user_repr, &pos_it_repr, rightfm.no_components);

            for i in (0..rightfm.max_sampled)
            {
                let negative_item_id = random_state.thread_rng.gen_range(0..item_features.nrows()) as usize;
                let neg_item_feature = item_features.get_row(negative_item_id).unwrap();
                compute_representation(&neg_item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut neg_it_repr);
                let neg_prediction = compute_prediction_from_repr(&user_repr, &neg_it_repr, rightfm.no_components);

                if (neg_prediction - pos_prediction) > 1. {
                    if interactions.get_row(user_id).unwrap().col_indices().binary_search(&negative_item_id).is_err() {
                        let mut _i = ((no_examples as Flt - 1.) / ((i + 1) as Flt)).floor();
                        _i = if _i < 1.0 { 1.0 } else { _i };
                        let mut loss = weight * _i.ln();
                        loss = if loss > MAX_LOSS { MAX_LOSS } else { loss };
                        warp_update(loss, &pos_item_feature, &neg_item_feature, &user_feature, &user_repr, &pos_it_repr, &neg_it_repr, rightfm, item_alpha, user_alpha);
                    }
                }
            }
            locked_regularize(rightfm, item_alpha, user_alpha);
        }
    });
    regularize(rightfm, item_alpha, user_alpha);
}


fn fit_bpr(
    item_features: &CsrMatrix<Flt>,
    user_features: &CsrMatrix<Flt>,
    interactions: &CsrMatrix<Flt>,
    user_ids: &[usize],
    item_ids: &[usize],
    values: &[Flt],
    sample_weight: &Vec<Flt>,
    shuffle_indices: &[usize],
    rightfm: &mut FastRightFM,
    learning_rate: Flt,
    item_alpha: Flt,
    user_alpha: Flt,
    random_state: &mut RandomState) {
    let no_examples = values.len();
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut pos_it_repr = vec![0.0; rightfm.no_components + 1];
    let mut neg_it_repr = vec![0.0; rightfm.no_components + 1];

    (0..no_examples).for_each(|i| {
        let row = shuffle_indices[i];
        if values[row] > 0.0 {
            let weight = sample_weight[row];
            let user_id = user_ids[row];
            let pos_item_id = item_ids[row];
            let mut idx = 0;
            let neg_item_id = loop {
                idx += 1;
                let _nii = item_ids[random_state.thread_rng.gen_range(0..no_examples)];
                if idx >= no_examples || interactions.get_row(user_id).unwrap().col_indices().binary_search(&_nii).is_err() {
                    break _nii;
                }
            };
            let neg_item_feature = item_features.get_row(neg_item_id).unwrap();
            let pos_item_feature = item_features.get_row(pos_item_id).unwrap();
            let user_feature = user_features.get_row(user_id).unwrap();
            compute_representation(&pos_item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut pos_it_repr);
            compute_representation(&user_feature, &rightfm.user_features, &rightfm.user_biases, rightfm, rightfm.user_scale, &mut user_repr);
            compute_representation(&neg_item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut neg_it_repr);
            let pos_prediction = compute_prediction_from_repr(&user_repr, &pos_it_repr, rightfm.no_components);
            let neg_prediction = compute_prediction_from_repr(&user_repr, &neg_it_repr, rightfm.no_components);
            let loss = weight * (1.0 - sigmoid(pos_prediction - neg_prediction));
            warp_update(loss, &pos_item_feature, &neg_item_feature, &user_feature, &user_repr, &pos_it_repr, &neg_it_repr, rightfm, item_alpha, user_alpha);
            locked_regularize(rightfm, item_alpha, user_alpha);
        }
    });
    regularize(rightfm, item_alpha, user_alpha);
}

fn fit_warp_kos(
    item_features: &CsrMatrix<Flt>,
    user_features: &CsrMatrix<Flt>,
    interactions: &CsrMatrix<Flt>,
    user_ids: &[usize],
    shuffle_indices: &[usize],
    rightfm: &mut FastRightFM,
    learning_rate: Flt,
    item_alpha: Flt,
    user_alpha: Flt,
    k: usize,
    n: usize,
    random_state: &mut RandomState) {
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut pos_it_repr = vec![0.0; rightfm.no_components + 1];
    let mut neg_it_repr = vec![0.0; rightfm.no_components + 1];
    let mut pos_pairs = Vec::with_capacity(n);

    for (i, &row) in shuffle_indices.iter().enumerate() {
        let user_id = user_ids[row];
        let user_feature = user_features.get_row(user_id).unwrap();
        compute_representation(&user_feature, &rightfm.user_features, &rightfm.user_biases, rightfm, rightfm.user_scale, &mut user_repr);

        let interaction = interactions.get_row(user_id).unwrap();
        if interaction.nnz() > 0 {
            let no_positives = if interaction.nnz() > n { n } else { interaction.nnz() };
            (0..no_positives).for_each(|i| {
                let sampled_positive_item_id = random_state.thread_rng.gen_range(0..no_positives);
                let item_feature = item_features.get_row(sampled_positive_item_id).unwrap();
                compute_representation(&item_feature, &rightfm.item_features, &rightfm.item_biases, &rightfm, rightfm.item_scale, &mut pos_it_repr);
                let sampled_positive_prediction = compute_prediction_from_repr(&user_repr, &pos_it_repr, rightfm.no_components);
                pos_pairs.push((sampled_positive_item_id, sampled_positive_prediction));
            });
            pos_pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let (pos_item_id, pos_prediction) = pos_pairs[cmp::min(k, no_positives) - 1];
            let pos_item_feature = item_features.get_row(pos_item_id).unwrap();
            compute_representation(&pos_item_feature, &rightfm.item_features, &rightfm.item_biases, &rightfm, rightfm.item_scale, &mut pos_it_repr);
            for i in (0..rightfm.max_sampled) {
                let neg_item_id = random_state.thread_rng.gen_range(0..item_features.nrows()) as usize;
                let neg_item_feature = item_features.get_row(neg_item_id).unwrap();
                compute_representation(&neg_item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut neg_it_repr);
                let neg_prediction = compute_prediction_from_repr(&user_repr, &neg_it_repr, rightfm.no_components);
                if (neg_prediction - pos_prediction) > 1. {
                    if interactions.get_row(user_id).unwrap().col_indices().binary_search(&neg_item_id).is_err() {
                        let mut loss = Flt::ln(cmp::max(1, item_features.nrows() - 1) as Flt / ((i + 1) as Flt));
                        loss = if loss > MAX_LOSS { MAX_LOSS } else { loss };
                        warp_update(loss, &pos_item_feature, &neg_item_feature, &user_feature, &user_repr, &pos_it_repr, &neg_it_repr, rightfm, item_alpha, user_alpha);
                        break;
                    }
                }
            }
            locked_regularize(rightfm, item_alpha, user_alpha);
        }
    }
    regularize(rightfm, item_alpha, user_alpha);
}

fn predict_rightfm(
    user_features: &CsrMatrix<Flt>,
    item_features: &CsrMatrix<Flt>,
    user_ids: &[usize],
    item_ids: &[usize],
    predictions: &Array2<Flt>,
    rightfm: &FastRightFM,
    num_threads: usize,
) {
    let no_examples = predictions.shape()[0];
    let component_size = (rightfm.no_components + 1) as usize;
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut it_repr = vec![0.0; rightfm.no_components + 1];
    for i in 0..no_examples {
        let user_feature = user_features.get_row(user_ids[i]).unwrap();
        compute_representation(
            &user_feature,
            &rightfm.user_features,
            &rightfm.user_biases,
            rightfm,
            rightfm.user_scale,
            &mut user_repr,
        );
        let item_feature = item_features.get_row(item_ids[i]).unwrap();
        compute_representation(
            &item_feature,
            &rightfm.item_features,
            &rightfm.item_biases,
            rightfm,
            rightfm.item_scale,
            &mut it_repr,
        );
        predictions[[i,i]] = compute_prediction_from_repr(&user_repr, &it_repr, rightfm.no_components);
    }
}

fn predict_ranks(
    user_features: &CsrMatrix<Flt>,
    item_features: &CsrMatrix<Flt>,
    test_interactions: &CsrMatrix<Flt>,
    train_interactions: &CsrMatrix<Flt>,
    ranks: &mut [Flt],
    rightfm: &FastRightFM,
    num_threads: usize,
) {
    let mut predictions_size = 0;
    for user_id in 0..test_interactions.nrows() {
        predictions_size = predictions_size.max(test_interactions.get_row(user_id).unwrap().nnz());
    }
    let mut user_repr = vec![0.0; rightfm.no_components + 1];
    let mut it_repr = vec![0.0; rightfm.no_components + 1];
    let mut predictions = vec![0.0; predictions_size];
    let mut item_ids = vec![0usize; predictions_size];

    for user_id in 0..test_interactions.nrows() {
        let test_interaction = test_interactions.get_row(user_id).unwrap();
        if (test_interaction.nnz() > 0) {
            let user_feature = user_features.get_row(user_id).unwrap();
            compute_representation(&user_feature, &rightfm.user_features, &rightfm.user_biases, rightfm, rightfm.user_scale, &mut user_repr);
            for i in 0..test_interaction.nnz() {
                let item_id = test_interaction.col_indices()[i];
                let item_feature = item_features.get_row(item_id).unwrap();
                compute_representation(&item_feature, &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut it_repr);
                item_ids[i] = item_id;
                predictions[i] = compute_prediction_from_repr(&user_repr, &it_repr, rightfm.no_components);
            }
            for item_id in 0..test_interactions.ncols() {
                if train_interactions.get_row(user_id).unwrap().col_indices().binary_search(&item_id).is_err() {
                    compute_representation(&item_features.get_row(item_id).unwrap(), &rightfm.item_features, &rightfm.item_biases, rightfm, rightfm.item_scale, &mut it_repr);
                    let prediction = compute_prediction_from_repr(&user_repr, &it_repr, rightfm.no_components);
                    for i in 0..test_interaction.nnz() {
                        if item_id != item_ids[i] && prediction >= predictions[i] {
                            ranks[test_interactions.row_offsets()[user_id] + i] += 1.0;
                        }
                    }
                }
            }
        }
    }
}

fn calculate_auc_from_rank(
    ranks: &CsrMatrix<Flt>,
    num_train_positives: &[usize],
    rank_data: &mut [Flt],
    auc: &mut [Flt],
    num_threads: usize,
) {
    for i in 0..ranks.nrows() {
        let rank = ranks.get_row(i).unwrap();
        let num_positives = rank.nnz();
        let num_negatives = rank.ncols() - (rank.nnz() + num_train_positives[i]);
        if num_positives == 0 || num_negatives == ranks.ncols() {
            auc[i] = 0.5;
        } else {
            partial_sort(rank_data, ranks.row_offsets()[i], ranks.row_offsets()[i + 1], true);
            for j in 0..num_positives {
                let rank = if rank_data[ranks.row_offsets()[i] + j] < j as Flt {0.} else {rank_data[ranks.row_offsets()[i] + j] - j as Flt};
                auc[i] += 1. - rank/num_negatives as Flt;
            }

            if num_positives >0 {
                auc[i] /= num_positives as Flt;
            }
        }
    }
}

#[inline(always)]
fn in_positives(item_id: usize, interactions: &CsrRow<Flt>) -> bool {
    interactions.col_indices().binary_search(&item_id).is_ok()
}

#[inline(always)]
fn compute_representation(
    features: &CsrRow<Flt>,
    feature_embeddings: &Array2<Flt>,
    feature_biases: &Array1<Flt>,
    rightfm: &FastRightFM,
    scale: Flt,
    representation: &mut Vec<Flt>,
) {
    // let row = features.get_row(row_id).unwrap();
    let indices = features.col_indices();
    for i in 0..=rightfm.no_components {
        representation[i] = 0.0;
    }

    for i in 0..indices.len() {
        let feature = indices[i];
        let feature_weight = features.values()[i] * scale;
        feature_embeddings.slice(s![feature, ..]).iter().enumerate().for_each(
            |(j, embedded_val)| representation[j] += feature_weight * embedded_val
        );
        representation[rightfm.no_components] += feature_weight * feature_biases[feature];
    }
}

#[inline(always)]
fn compute_prediction_from_repr(user_repr: &[Flt], item_repr: &[Flt], no_components: usize) -> Flt {
    let mut result = user_repr[no_components] + item_repr[no_components];

    // Latent factor dot product
    let user_repr_iter = &user_repr[..no_components];
    let item_repr_iter = &item_repr[..no_components];
    for (&u, &i) in user_repr_iter.iter().zip(item_repr_iter.iter()) {
        result += u * i;
    }
    result
}

#[inline(always)]
fn regularize(rightfm: &mut FastRightFM, item_alpha: Flt, user_alpha: Flt) {
    let no_features = rightfm.item_features.shape()[0];
    let no_users = rightfm.user_features.shape()[0];

    for i in 0..no_features {
        for j in 0..rightfm.no_components {
            rightfm.item_features[[i, j]] /= rightfm.item_scale;
        }
        rightfm.item_biases[i] /= rightfm.item_scale;
    }

    for i in 0..no_users {
        for j in 0..rightfm.no_components {
            rightfm.user_features[[i, j]] /= rightfm.user_scale;
        }
        rightfm.user_biases[i] /= rightfm.user_scale;
    }

    rightfm.item_scale = 1.0;
    rightfm.user_scale = 1.0;
}

#[inline(always)]
fn locked_regularize(rightfm: &mut FastRightFM, item_alpha: Flt, user_alpha: Flt) {
    if rightfm.item_scale > MAX_REG_SCALE || rightfm.user_scale > MAX_REG_SCALE {
        regularize(rightfm, item_alpha, user_alpha);
    }
}

#[inline(always)]
fn partial_sort(slice: &mut [Flt], start_idx: usize, end_idx: usize, reverse: bool) {
    let mut s = slice[start_idx..end_idx].to_vec();
    s.sort_unstable_by(|a, b| if reverse { b.partial_cmp(a).unwrap() } else { a.partial_cmp(b).unwrap() });
    for i in start_idx..end_idx {
        slice[i] = s[i - start_idx];
    }
    println!("{:?}", slice);
}

#[inline(always)]
fn update(loss: Flt,
          item_features: &CsrRow<Flt>,
          user_features: &CsrRow<Flt>,
          user_repr: &[Flt],
          it_repr: &[Flt],
          rightfm: &mut FastRightFM,
          item_alpha: Flt,
          user_alpha: Flt) {
    let no_components = rightfm.no_components as usize;
    let no_features = item_features.col_indices().len();
    let mut avg_learning_rate = 0.0;

    // Call hypothetical `update_biases` functions and accumulate their results.
    avg_learning_rate += update_biases(item_features,
                                       &mut rightfm.item_biases, &mut rightfm.item_biases_gradients,
                                       &mut rightfm.item_biases_momentum,
                                       loss,
                                       rightfm.learning_schedule.clone(),
                                       rightfm.learning_rate,
                                       item_alpha,
                                       rightfm.rho,
                                       rightfm.eps as Flt);

    avg_learning_rate += update_biases(user_features,
                                       &mut rightfm.user_biases, &mut rightfm.user_biases_gradients,
                                       &mut rightfm.user_biases_momentum,
                                       loss,
                                       rightfm.learning_schedule.clone(),
                                       rightfm.learning_rate,
                                       user_alpha,
                                       rightfm.rho,
                                       rightfm.eps as Flt);

    // Update latent representations.
    for i in 0..no_components {
        let user_component = user_repr[i];
        let item_component = it_repr[i];

        avg_learning_rate += update_features(item_features, &mut rightfm.item_features,
                                             &mut rightfm.item_features_gradients,
                                             &mut rightfm.item_features_momentum,
                                             i,
                                             loss * user_component,
                                             rightfm.learning_schedule.clone(),
                                             rightfm.learning_rate,
                                             item_alpha,
                                             rightfm.rho,
                                             rightfm.eps as Flt);

        avg_learning_rate += update_features(user_features, &mut rightfm.user_features,
                                             &mut rightfm.user_features_gradients,
                                             &mut rightfm.user_features_momentum,
                                             i,
                                             loss * item_component,
                                             rightfm.learning_schedule.clone(),
                                             rightfm.learning_rate,
                                             user_alpha,
                                             rightfm.rho,
                                             rightfm.eps as Flt);
    }

    // Normalize the average learning rate.
    avg_learning_rate /= ((no_components + 1) * user_features.nnz()
        + (no_components + 1) * item_features.nnz()) as Flt;

    // Update the scaling factors for lazy regularization.
    rightfm.item_scale *= 1.0 + item_alpha * avg_learning_rate;
    rightfm.user_scale *= 1.0 + user_alpha * avg_learning_rate;
}

fn warp_update(
    loss: Flt,
    pos_item_features: &CsrRow<Flt>,
    neg_item_features: &CsrRow<Flt>,
    user_features: &CsrRow<Flt>,
    user_repr: &[Flt],
    pos_it_repr: &[Flt],
    neg_it_repr: &[Flt],
    rightfm: &mut FastRightFM,
    item_alpha: Flt,
    user_alpha: Flt,
) {
    let mut avg_learning_rate = 0.0;

    avg_learning_rate += update_biases(
        pos_item_features,
        &mut rightfm.item_biases,
        &mut rightfm.item_biases_gradients,
        &mut rightfm.item_biases_momentum,
        -loss,
        rightfm.learning_schedule.clone(),
        rightfm.learning_rate,
        item_alpha,
        rightfm.rho,
        rightfm.eps as Flt,
    );

    avg_learning_rate += update_biases(
        neg_item_features,
        &mut rightfm.item_biases,
        &mut rightfm.item_biases_gradients,
        &mut rightfm.item_biases_momentum,
        loss,
        rightfm.learning_schedule.clone(),
        rightfm.learning_rate,
        item_alpha,
        rightfm.rho,
        rightfm.eps as Flt,
    );

    avg_learning_rate += update_biases(
        user_features,
        &mut rightfm.user_biases,
        &mut rightfm.user_biases_gradients,
        &mut rightfm.user_biases_momentum,
        loss,
        rightfm.learning_schedule.clone(),
        rightfm.learning_rate,
        user_alpha,
        rightfm.rho,
        rightfm.eps as Flt,
    );

    for i in 0..rightfm.no_components {
        let user_component = user_repr[i];
        let positive_item_component = pos_it_repr[i];
        let negative_item_component = neg_it_repr[i];

        avg_learning_rate += update_features(
            pos_item_features,
            &mut rightfm.item_features,
            &mut rightfm.item_features_gradients,
            &mut rightfm.item_features_momentum,
            i,
            -loss * user_component,
            rightfm.learning_schedule.clone(),
            rightfm.learning_rate,
            item_alpha,
            rightfm.rho,
            rightfm.eps as Flt,
        );

        avg_learning_rate += update_features(
            neg_item_features,
            &mut rightfm.item_features,
            &mut rightfm.item_features_gradients,
            &mut rightfm.item_features_momentum,
            i,
            loss * user_component,
            rightfm.learning_schedule.clone(),
            rightfm.learning_rate,
            item_alpha,
            rightfm.rho,
            rightfm.eps as Flt,
        );

        avg_learning_rate += update_features(
            user_features,
            &mut rightfm.user_features,
            &mut rightfm.user_features_gradients,
            &mut rightfm.user_features_momentum,
            i,
            loss * (negative_item_component - positive_item_component),
            rightfm.learning_schedule.clone(),
            rightfm.learning_rate,
            user_alpha,
            rightfm.rho,
            rightfm.eps as Flt,
        );
    }

    avg_learning_rate /= ((rightfm.no_components + 1) * user_features.nnz()
        + (rightfm.no_components + 1) * pos_item_features.nnz()
        + (rightfm.no_components + 1) * neg_item_features.nnz()) as Flt;

    rightfm.item_scale *= (1.0 + item_alpha * avg_learning_rate);
    rightfm.user_scale *= (1.0 + user_alpha * avg_learning_rate);
}

#[inline(always)]
fn update_biases(feature_indices: &CsrRow<Flt>,
                 biases: &mut Array1<Flt>,
                 gradients: &mut Array1<Flt>,
                 momentum: &mut Array1<Flt>,
                 gradient: Flt,
                 learning_schedule: LearningSchedule,
                 learning_rate: Flt,
                 alpha: Flt,
                 rho: Flt,
                 eps: Flt) -> Flt {
    let mut sum_learning_rate = 0.0;
    let range = (0..feature_indices.col_indices().len());
    match learning_schedule {
        LearningSchedule::AdaDelta => {
            for i in range {
                let feature = feature_indices.col_indices()[i];
                let feature_weight = feature_indices.values()[i];

                gradients[feature] = rho * gradients[feature] + (1.0 - rho) * (feature_weight * gradient).powi(2);
                let local_learning_rate = (momentum[feature].sqrt() + eps).recip() * gradients[feature].sqrt();
                let update = local_learning_rate * gradient * feature_weight;
                momentum[feature] = rho * momentum[feature] + (1.0 - rho) * update.powi(2);
                biases[feature] -= update;

                // Lazy regularization: scale up by the regularization parameter.
                biases[feature] *= 1.0 + alpha * local_learning_rate;

                sum_learning_rate += local_learning_rate;
            }
        }
        LearningSchedule::AdaGrad => {
            for i in range {
                let feature = feature_indices.col_indices()[i];
                let feature_weight = feature_indices.values()[i];

                let local_learning_rate = learning_rate / gradients[feature].sqrt();
                biases[feature] -= local_learning_rate * feature_weight * gradient;
                gradients[feature] += (gradient * feature_weight).powi(2);

                // Lazy regularization: scale up by the regularization parameter.
                biases[feature] *= 1.0 + alpha * local_learning_rate;

                sum_learning_rate += local_learning_rate;
            }
        }
    }
    sum_learning_rate
}

#[inline(always)]
fn update_features(feature_indices: &CsrRow<Flt>,
                   features: &mut Array2<Flt>,
                   gradients: &mut Array2<Flt>,
                   momentum: &mut Array2<Flt>,
                   component: usize,
                   gradient: Flt,
                   learning_schedule: LearningSchedule,
                   learning_rate: Flt,
                   alpha: Flt,
                   rho: Flt,
                   eps: Flt) -> Flt {
    let mut sum_learning_rate = 0.0;
    let range = (0..feature_indices.col_indices().len());
    match learning_schedule {
        LearningSchedule::AdaDelta => {
            for i in range {
                let feature = feature_indices.col_indices()[i];
                let feature_weight = feature_indices.values()[i];

                gradients[[feature, component]] = rho * gradients[[feature, component]] + (1.0 - rho) * (feature_weight * gradient).powi(2);
                let local_learning_rate = (momentum[[feature, component]].sqrt() + eps).recip() * gradients[[feature, component]].sqrt();
                let update = local_learning_rate * gradient * feature_weight;
                momentum[[feature, component]] = rho * momentum[[feature, component]] + (1.0 - rho) * update.powi(2);
                features[[feature, component]] -= update;

                // Lazy regularization: scale up by the regularization parameter.
                features[[feature, component]] *= 1.0 + alpha * local_learning_rate;

                sum_learning_rate += local_learning_rate;
            }
        }
        LearningSchedule::AdaGrad => {
            for i in range {
                let feature = feature_indices.col_indices()[i] as usize;
                let feature_weight = feature_indices.values()[i];

                let local_learning_rate = learning_rate / gradients[[feature, component]].sqrt();
                features[[feature, component]] -= local_learning_rate * feature_weight * gradient;
                gradients[[feature, component]] += (gradient * feature_weight).powi(2);

                // Lazy regularization: scale up by the regularization parameter.
                features[[feature, component]] *= 1.0 + alpha * local_learning_rate;

                sum_learning_rate += local_learning_rate;
            }
        }
    }
    sum_learning_rate
}

fn sigmoid(x: Flt) -> Flt {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    #[test]
    fn test_fast_rightfm() {
        let no_item_features = 10;
        let no_components = 5;
        let data = Array2::from_elem((no_item_features, no_components), 0.5f32);
        println!("{}", data);
    }
}

