use ndarray::Array;
use pyo3::prelude::*;

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn};
use tokenizers_python::tokenizer::PyTokenizer;


/// Encodes text using a tokenizer from file
#[pyfunction]
fn encode<'py>(py: Python<'py>, tokenizer: &PyTokenizer, text: &str) -> PyResult<Bound<'py, PyArrayDyn<i64>>> {
    // Load tokenizer from file and perform tokenization in thread-safe manner
    let array_result = py.allow_threads(|| {
        // Perform tokenization
        match tokenizer.tokenizer.encode(text, false) {
            Ok(encoding) => {
                // Get IDs and convert to ndarray
                let ids = encoding.get_ids();
                let array: ArrayD<i64> = Array::from_shape_vec(
                    vec![ids.len()],
                    ids.into_iter().map(|id| *id as i64).collect(),
                ).expect("Failed to create ndarray from ids");
                
                Ok(array)
            },
            Err(e) => Err(format!("Failed to encode text: {}", e))
        }
    });
    
    // Convert any error to PyResult and return the array
    let array = array_result.map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;
    
    // Convert to Python array
    Ok(array.into_pyarray(py))
}

#[pymodule]
fn native_tokenizer_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    Ok(())
}