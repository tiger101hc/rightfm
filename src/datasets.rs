use crate::rightfm::Flt;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use polars::prelude::{CsvReader, DataType, SerReader};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct MovieLensData {
    pub train: CooMatrix<Flt>,
    pub test: CooMatrix<Flt>,
    pub item_features: CsrMatrix<Flt>,
    // item_feature_labels: Vec<String>,
    // item_labels: Vec<String>,
}

pub fn fetch_movielens(
    data_home: Option<&str>,
    indicator_features: bool,
    genre_features: bool,
    min_rating: f32,
    download_if_missing: bool,
) -> MovieLensData {
    assert!(
        indicator_features || genre_features,
        "At least one of item_indicator_features or genre_features must be True"
    );
    let data_home = if data_home.is_none() {
        String::from(shellexpand::tilde("~/rightfm/data"))
    } else {
        String::from(data_home.unwrap())
    };
    get_data(
        &data_home,
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "movielens.zip",
        download_if_missing,
    )
    .expect("download fail");
    extract_zip(&data_home, "/movielens.zip").unwrap();
    let data_type = vec![DataType::UInt32, DataType::UInt32, DataType::Float32];
    let data_home = data_home + "/movielens";
    let (train, test, _, num_items) = parse_to_interaction(
        &(data_home.clone() + "/ml-100k/ua.base"),
        &(data_home.clone() + "/ml-100k/ua.test"),
        &data_type,
        min_rating,
    );
    MovieLensData {
        train,
        test,
        item_features: parse_features(
            &(data_home.clone() + "/ml-100k/u.item"),
            if genre_features {
                Some(data_home.clone() + "/ml-100k/u.genre")
            } else {
                None
            },
            num_items,
            indicator_features,
        ),
    }
}

fn extract_zip(data_home: &str, zip_path: &str) -> std::io::Result<()> {
    let zip_file = std::fs::File::open(data_home.to_string() + zip_path).unwrap();
    let mut zip = zip::ZipArchive::new(zip_file).unwrap();
    let data_dir = Path::new(data_home).canonicalize()?;
    if !data_dir.join("movielens").exists() {
        zip.extract(data_dir.join("movielens"))
            .expect("unzip occur error");
    }
    Ok(())
}

fn parse_to_interaction(
    train_file: &str,
    test_file: &str,
    data_type: &[DataType],
    min_rating: Flt,
) -> (CooMatrix<Flt>, CooMatrix<Flt>, usize, usize) {
    let mut df1 = CsvReader::from_path(train_file)
        .unwrap()
        .with_separator(9)
        .has_header(false)
        .with_projection(Some(vec![0, 1, 2]))
        .with_dtypes_slice(Some(data_type))
        .finish()
        .unwrap();
    let mut df2 = CsvReader::from_path(test_file)
        .unwrap()
        .with_separator(9)
        .has_header(false)
        .with_projection(Some(vec![0, 1, 2]))
        .with_dtypes_slice(Some(data_type))
        .finish()
        .unwrap();
    let no_user1: usize = df1[0].u32().unwrap().iter().max().unwrap().unwrap() as usize;
    let no_item1: usize = df1[1].u32().unwrap().iter().max().unwrap().unwrap() as usize;
    let no_user2: usize = df1[0].u32().unwrap().iter().max().unwrap().unwrap() as usize;
    let no_item2: usize = df1[1].u32().unwrap().iter().max().unwrap().unwrap() as usize;
    let no_users = no_user1.max(no_user2);
    let no_items = no_item1.max(no_item2);

    df1.apply_at_idx(0, |s| s - 1).expect("");
    df1.apply_at_idx(1, |s| s - 1).expect("");
    df2.apply_at_idx(0, |s| s - 1).expect("");
    df2.apply_at_idx(1, |s| s - 1).expect("");

    let mut coo1 = CooMatrix::new(no_users, no_items);
    let mut coo2 = CooMatrix::new(no_users, no_items);
    println!(
        "test shape:{:?},train shape:{:?}",
        (coo2.nrows(), coo2.ncols()),
        (coo1.nrows(), coo1.ncols())
    );

    for i in 0..df1.height().max(df2.height()) {
        if i < df1.height() {
            let s = &df1.get_row(i).unwrap().0;
            if s[2].try_extract::<Flt>().unwrap() >= min_rating {
                coo1.push(
                    s[0].try_extract().unwrap(),
                    s[1].try_extract().unwrap(),
                    s[2].try_extract::<Flt>().unwrap(),
                )
            }
        }
        if i < df2.height() {
            let s = &df2.get_row(i).unwrap().0;
            if s[2].try_extract::<Flt>().unwrap() >= min_rating {
                coo2.push(
                    s[0].try_extract().unwrap(),
                    s[1].try_extract().unwrap(),
                    s[2].try_extract::<Flt>().unwrap(),
                )
            }
        }
    }
    (coo1, coo2, no_users, no_items)
}

fn parse_features(
    item: &str,
    genre: Option<String>,
    no_items: usize,
    indicator_features: bool,
) -> CsrMatrix<Flt> {
    let genre_matrix = if genre.is_some() {
        let g_type = vec![DataType::String, DataType::UInt32];
        let dfg = CsvReader::from_path(genre.unwrap())
            .unwrap()
            .with_separator('|' as u8)
            .has_header(false)
            .with_dtypes_slice(Some(&g_type))
            .finish()
            .unwrap();
        let no_features = dfg[1].u32().unwrap().iter().max().unwrap().unwrap() as usize;
        let i_type = vec![
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
        ];
        let mut dfi = CsvReader::from_path(item)
            .unwrap()
            .with_separator('|' as u8)
            .has_header(false)
            .with_projection(Some(vec![
                0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ]))
            .with_dtypes_slice(Some(&i_type))
            .finish()
            .unwrap();
        let _no_item: usize = dfi[0].u32().unwrap().iter().max().unwrap().unwrap() as usize;
        assert!(no_items >= _no_item, "train data should cover item list");
        let mut coo = CooMatrix::new(no_items, no_features);
        dfi.apply_at_idx(0, |s| s - 1).expect("panic at update idx");
        for i in 0..dfi.height() {
            let s = &dfi.get_row(i).unwrap().0;
            let mut i = 0;
            s.iter().enumerate().for_each(|(j, s)| {
                let v = s.try_extract::<usize>().unwrap();
                if j == 0 {
                    i = v;
                } else {
                    if v > 0 {
                        coo.push(i, j - 1, 1.0 as Flt);
                    }
                }
            })
        }
        Some(coo)
    } else {
        None
    };
    let idf = if indicator_features {
        Some(
            CooMatrix::try_from_triplets(
                no_items,
                no_items,
                (0..no_items).collect::<Vec<usize>>(),
                (0..no_items).collect::<Vec<usize>>(),
                vec![1 as Flt; no_items],
            )
            .unwrap(),
        )
    } else {
        None
    };

    match (genre_matrix, idf) {
        (Some(g), Some(i)) => CsrMatrix::from(
            &CooMatrix::try_from_triplets(
                no_items,
                g.ncols() + i.ncols(),
                vec![g.row_indices().to_vec(), i.row_indices().to_vec()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<usize>>(),
                vec![g.col_indices().to_vec(), i.col_indices().to_vec()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<usize>>(),
                vec![g.values().to_vec(), i.values().to_vec()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Flt>>(),
            )
            .unwrap(),
        ),
        (Some(g), None) => CsrMatrix::from(&g),
        (None, Some(i)) => CsrMatrix::from(&i),
        (None, None) => panic!("should not happen"),
    }
}

fn get_data(
    data_home: &str,
    url: &str,
    dest_filename: &str,
    download_if_missing: bool,
) -> Result<PathBuf, std::io::Error> {
    let data_dir = Path::new(data_home).canonicalize()?;
    let dest_path = data_dir.join(dest_filename);
    if !dest_path.is_file() {
        if download_if_missing {
            download(url, &dest_path)?;
        } else {
            return Err(Error::new(ErrorKind::NotFound, "Dataset missing."));
        }
    }
    Ok(dest_path)
}

fn download(url: &str, dest_path: &Path) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(dest_path)?;
    let mut response = reqwest::blocking::get(url).expect("download failed");
    std::io::copy(&mut response, &mut file).expect("download failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fetch_movielens() {
        let md = super::fetch_movielens(None, false, true, 1.0, true);
        println!("{:?},{:?},{:?}", md.item_features, md.test, md.train);
    }
}
