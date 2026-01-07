// TODO: Parallelize this
pub fn flat_matrix_vector_dot(matrix: &[f32], vec: &[f32], start_idx: usize) -> Result<f32, &'static str> {
    let mut res: f32 = 0.0;

    for i in 0..vec.len() {
        if start_idx + i >= matrix.len() {
            return Err("Index out of bounds of the matrix!");
        }

        res += vec[i] * matrix[start_idx + i];
    }

    Ok(res)
}