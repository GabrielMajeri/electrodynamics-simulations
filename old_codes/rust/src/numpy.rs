use std::{fs::File, io::BufWriter, path::Path};

use nalgebra::{ArrayStorage, Vector};
use npyz::{Serialize, WriterBuilder};

use crate::types::Real;

/// Save the given array of vectors to disk in the NPY binary file format.
pub fn write_to_npy_file<P: AsRef<Path>, T: Copy + Serialize, D, const CD: usize>(
    path: P,
    data: &Vec<Vector<T, D, ArrayStorage<T, CD, 1>>>,
) {
    let file = File::create(path).expect("Failed to open .npy file for writing");
    let mut writer = BufWriter::new(file);

    let type_string = format!("<f{}", std::mem::size_of::<Real>())
        .parse::<npyz::TypeStr>()
        .unwrap();
    let dtype = npyz::DType::Plain(type_string);

    let mut writer = npyz::WriteOptions::<T>::new()
        .dtype(dtype)
        .shape(&[data.len().try_into().unwrap(), CD as u64])
        .writer(&mut writer)
        .begin_nd()
        .expect("Failed to initialize NPY writer");

    let flattened = data
        .iter()
        .flat_map(|v| v.data.0.iter().flat_map(|r| r.iter().copied()));

    writer
        .extend(flattened)
        .expect("Failed to write array to file");

    writer.finish().expect("Failed to finish writing NPY file");
}
