use std::str::FromStr;

use crate::rightfm_fast::{
    fit_bpr, fit_logistic, fit_warp, fit_warp_kos, predict_ranks, predict_rightfm, FastRightFM,
};
use assert_ok::assert_ok;
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use rand::prelude::ThreadRng;
use shuffle::fy::FisherYates;
use shuffle::shuffler::Shuffler;
use tqdm::Iter;

pub(crate) type Flt = f32;

#[derive(Eq, PartialEq, Clone)]
pub enum LearningSchedule {
    AdaGrad,
    AdaDelta,
}

impl FromStr for LearningSchedule {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "adagrad" => Ok(LearningSchedule::AdaGrad),
            "adadelta" => Ok(LearningSchedule::AdaDelta),
            _ => Err(format!("Invalid learning schedule: {}", s)),
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
pub enum Loss {
    Logistic,
    Warp,
    Bpr,
    WarpKos,
}

impl FromStr for Loss {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "logistic" => Ok(Loss::Logistic),
            "warp" => Ok(Loss::Warp),
            "bpr" => Ok(Loss::Bpr),
            "warpkos" => Ok(Loss::WarpKos),
            _ => Err(format!("Invalid loss type: {}", s)),
        }
    }
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
    pub(crate) fy: FisherYates,
    pub(crate) thread_rng: ThreadRng,
}

impl RandomState {
    pub fn default() -> RandomState {
        RandomState {
            fy: FisherYates::default(),
            thread_rng: rand::thread_rng(),
        }
    }
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

    fn reset_state(&mut self) {
        self.data = None;
    }

    pub fn fit(
        &mut self,
        interaction: &CooMatrix<Flt>,
        user_features: Option<CsrMatrix<Flt>>,
        item_features: Option<CsrMatrix<Flt>>,
        sample_weight: Option<&CooMatrix<Flt>>,
        epochs: usize,
        number_thread: u8,
        verbose: bool,
    ) {
        assert!(number_thread >= 1);
        self.reset_state();
        self.fit_partial(
            interaction,
            user_features,
            item_features,
            sample_weight,
            epochs,
            number_thread,
            verbose,
        );
    }

    pub fn fit_partial(
        &mut self,
        interaction: &CooMatrix<Flt>,
        user_features: Option<CsrMatrix<Flt>>,
        item_features: Option<CsrMatrix<Flt>>,
        sample_weight: Option<&CooMatrix<Flt>>,
        epochs: usize,
        number_thread: u8,
        verbose: bool,
    ) {
        let (no_user, no_item) = (interaction.nrows(), interaction.ncols());
        let _user_features = if user_features.is_some() {
            user_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(no_user)
        };
        let _item_features = if item_features.is_some() {
            item_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(no_item)
        };
        assert!(
            interaction.nrows() >= _user_features.nrows(),
            "interaction matrix must have more rows than user features matrix"
        );
        assert!(
            interaction.ncols() >= _item_features.nrows(),
            "interaction matrix must have more cols than item features matrix"
        );
        let sample_weight_data = self.process_sample_weight(interaction, sample_weight);
        if self.data.is_none() {
            self.data = Some(FastRightFM::new(
                self.no_components,
                _item_features.ncols(),
                _user_features.ncols(),
                self.learning_schedule.clone(),
                self.learning_rate,
                self.rho,
                self.epsilon,
                self.max_sampled,
            ));
        } else {
            assert_eq!(
                self.data.as_ref().unwrap().item_features.nrows(),
                _item_features.ncols(),
                "should have same no of item features"
            );
            assert_eq!(
                self.data.as_ref().unwrap().user_features.nrows(),
                _user_features.ncols(),
                "should have same no of user features"
            );
        }

        Self::process(epochs, verbose).for_each(|_| {
            self.run_epoch(
                &_item_features,
                &_user_features,
                interaction,
                &sample_weight_data,
                number_thread,
                self.loss.clone(),
            )
        });
    }

    fn run_epoch(
        &mut self,
        item_features: &CsrMatrix<Flt>,
        user_features: &CsrMatrix<Flt>,
        interaction: &CooMatrix<Flt>,
        sample_weight: &Vec<Flt>,
        number_thread: u8,
        loss: Loss,
    ) {
        let mut shuffled_indices = (0..interaction.nnz()).collect::<Vec<_>>();
        assert_ok!(self
            .random_state
            .fy
            .shuffle(&mut shuffled_indices, &mut self.random_state.thread_rng));
        match loss {
            Loss::Logistic => fit_logistic(
                item_features,
                user_features,
                interaction.row_indices(),
                interaction.col_indices(),
                interaction.values(),
                sample_weight,
                shuffled_indices.as_slice(),
                self.data.as_mut().unwrap(),
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                number_thread,
            ),
            Loss::Warp => {
                fit_warp(
                    item_features,
                    user_features,
                    &CsrMatrix::from(interaction),
                    interaction.row_indices(),
                    interaction.col_indices(),
                    interaction.values(),
                    sample_weight,
                    shuffled_indices.as_slice(),
                    self.data.as_mut().unwrap(),
                    self.learning_rate,
                    self.item_alpha,
                    self.user_alpha,
                    &mut self.random_state,
                );
            }
            Loss::Bpr => {
                fit_bpr(
                    item_features,
                    user_features,
                    &CsrMatrix::from(interaction),
                    interaction.row_indices(),
                    interaction.col_indices(),
                    interaction.values(),
                    sample_weight,
                    shuffled_indices.as_slice(),
                    self.data.as_mut().unwrap(),
                    self.learning_rate,
                    self.item_alpha,
                    self.user_alpha,
                    &mut self.random_state,
                );
            }
            Loss::WarpKos => {
                fit_warp_kos(
                    item_features,
                    user_features,
                    &CsrMatrix::from(interaction),
                    interaction.row_indices(),
                    shuffled_indices.as_slice(),
                    self.data.as_mut().unwrap(),
                    self.learning_rate,
                    self.item_alpha,
                    self.user_alpha,
                    self.k as usize,
                    self.n as usize,
                    &mut self.random_state,
                );
            }
        }
    }

    fn process(n: usize, verbose: bool) -> Box<dyn Iterator<Item = usize>> {
        if verbose {
            Box::new((0..n).tqdm())
        } else {
            Box::new(0..n)
        }
    }

    fn process_sample_weight(
        &self,
        interaction: &CooMatrix<Flt>,
        sample_weight: Option<&CooMatrix<Flt>>,
    ) -> Vec<Flt> {
        if sample_weight.is_some() {
            let data = sample_weight.unwrap();
            assert!(
                self.loss != Loss::WarpKos,
                "k-OS loss with sample weights not implemented."
            );
            assert!(
                data.row_indices() == interaction.row_indices()
                    && data.col_indices() == interaction.col_indices(),
                "sample weight and interactions matrices must be the same shape and same order"
            );
            return data.values().to_vec();
        }
        return DVector::repeat(interaction.nnz(), 1.).as_slice().to_vec();
    }

    pub fn predict(
        &mut self,
        user_ids: &[usize],
        item_ids: &[usize],
        user_features: Option<CsrMatrix<Flt>>,
        item_features: Option<CsrMatrix<Flt>>,
    ) -> Vec<Flt> {
        assert_eq!(
            user_ids.len(),
            item_ids.len(),
            "user_ids and item_ids must be the same length"
        );
        assert!(
            *user_ids.iter().min().unwrap() > 0usize && *item_ids.iter().min().unwrap() > 0usize,
            "user_ids and item_ids must be positive"
        );
        let no_users = user_ids.iter().max().unwrap() + 1;
        let no_items = item_ids.iter().max().unwrap() + 1;
        let _user_features = if user_features.is_some() {
            user_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(no_users)
        };
        let _item_features = if item_features.is_some() {
            item_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(no_items)
        };
        assert_eq!(
            no_users,
            _user_features.nrows(),
            "user_ids and user_features must have the same no of users"
        );
        assert_eq!(
            no_items,
            _item_features.nrows(),
            "item_ids and item_features must have the same no of items"
        );
        if self.data.is_none() {
            self.data = Some(FastRightFM::new(
                self.no_components,
                _item_features.ncols(),
                _user_features.ncols(),
                self.learning_schedule.clone(),
                self.learning_rate,
                self.rho,
                self.epsilon,
                self.max_sampled,
            ));
        }
        let _data = self.data.as_ref().unwrap();
        assert_eq!(
            _item_features.ncols(),
            _data.item_features.shape().0,
            "no. of item features must have the same shape[0] to item embedded"
        );
        assert_eq!(
            _user_features.ncols(),
            _data.user_features.shape().0,
            "no. of user features must have the same shape[0] to user embedded"
        );
        let mut predictions = vec![0.; user_ids.len()];
        predict_rightfm(
            &_user_features,
            &_item_features,
            user_ids,
            item_ids,
            &mut predictions,
            _data,
            1,
        );
        predictions
    }

    pub fn predict_rank(
        &self,
        test_interactions: &CsrMatrix<f32>,
        train_interactions: Option<CsrMatrix<f32>>,
        user_features: Option<CsrMatrix<f32>>,
        item_features: Option<CsrMatrix<f32>>,
        num_threads: usize,
        check_intersections: bool,
    ) -> CsrMatrix<Flt> {
        assert!(
            self.data.is_some(),
            "FastRightFM model must be fitted before calling predict_rank"
        );
        let mut _train_interactions = if check_intersections && train_interactions.is_some() {
            let _t = train_interactions.unwrap();
            let n_intersections = elementwise_multiply(&_t, &test_interactions).nnz();
            assert_eq!(
                n_intersections, 0,
                "train_interactions and test_interactions must not have any intersections"
            );
            Some(_t)
        } else {
            train_interactions
        };
        let (n_users, n_items) = (test_interactions.nrows(), test_interactions.ncols());
        let _user_features = if user_features.is_some() {
            user_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(n_users)
        };
        let _item_features = if item_features.is_some() {
            item_features.unwrap()
        } else {
            CsrMatrix::<Flt>::identity(n_items)
        };
        let _data = self.data.as_ref().unwrap();
        assert_eq!(
            _item_features.ncols(),
            _data.item_features.shape().0,
            "no. of item features must have the same shape[0] to item embedded"
        );
        assert_eq!(
            _user_features.ncols(),
            _data.user_features.shape().0,
            "no. of user features must have the same shape[0] to user embedded"
        );
        if _train_interactions.is_none() {
            _train_interactions = Some(CsrMatrix::zeros(n_users, n_items));
        }
        let mut ranks = vec![0.; test_interactions.nnz()];
        predict_ranks(
            &_user_features,
            &_item_features,
            &test_interactions,
            _train_interactions.as_ref().unwrap(),
            &mut ranks,
            _data,
            num_threads,
        );
        CsrMatrix::try_from_csr_data(
            n_users,
            n_items,
            test_interactions.row_offsets().to_vec(),
            test_interactions.col_indices().to_vec(),
            ranks,
        )
        .unwrap()
    }
}

fn elementwise_multiply(a: &CsrMatrix<Flt>, b: &CsrMatrix<Flt>) -> CsrMatrix<Flt> {
    assert!(
        a.nrows() == b.nrows() && a.ncols() == b.ncols(),
        "they must have same shape"
    );
    let mut coo = CooMatrix::new(a.nrows(), a.ncols());
    a.row_iter()
        .zip(b.row_iter())
        .enumerate()
        .for_each(|(idx, (ar, br))| {
            if ar.nnz() > 0 && br.nnz() > 0 {
                // let ab = BTreeSet::from_iter(ar.col_indices().to_vec());
                // let bb = BTreeSet::from_iter(br.col_indices().to_vec());
                // let intersection = ab.intersection(&bb).collect::<Vec<_>>();
                // if intersection.len() > 0 {
                //     intersection.iter().for_each(|&&i| {
                //         coo.push(idx, i, ar.values()[i] * br.values()[i]);
                //     });
                // }

                for (i, v) in br.col_indices().iter().enumerate() {
                    match ar.col_indices().binary_search(v) {
                        Ok(iii) => {
                            coo.push(idx, *v, ar.values()[iii] * br.values()[i]);
                        }
                        Err(_) => {}
                    }
                }
            }
        });
    CsrMatrix::from(&coo)
}
