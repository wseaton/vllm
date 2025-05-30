use ndarray::Array;
use pyo3::prelude::*;

use tokenizers_python::tokenizer::PyTokenizer;
use tokenizers::Tokenizer;
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};


/// Formats the sum of two numbers as string.
#[pyfunction]
fn encode<'py>(_py: Python<'py>, tokenizer: &PyTokenizer, text: String) -> PyResult<String> {
    let t = tokenizer.tokenizer.clone();
    let result = t.encode(text, false).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to encode text: {}",
            e
        ))
    }).map(|encoding| encoding.get_tokens().join(" "));
    Ok(result?)
}

#[pymodule]
fn native_tokenizer_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    Ok(())
}