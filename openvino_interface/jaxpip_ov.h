#ifndef JAXPIP_OV_H_
#define JAXPIP_OV_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize OpenVINO runtime and load the ONNX model.
 * @param model_path Path to the exported ONNX model file.
 */
void init_ov_model(const char* model_path);

/**
 * @brief Evaluate the potential energy and analytical forces via OpenVINO.
 * 
 * @param coords    Flattened 1D array of atomic coordinates [x1, y1, z1, ...].
 *                  Size must be 3 * num_atoms.
 * @param num_atoms Pointer to the number of atoms.
 * @param energy    Pointer to store output potential energy.
 * @param forces    Flattened 1D array to store output forces (3 * num_atoms).
 */
void eval_ov_model(double* coords, int* num_atoms, double* energy, double* forces);

/**
 * @brief Clean up and release OpenVINO resources.
 */
void finalize_ov_model();

#ifdef __cplusplus
}
#endif

#endif // JAXPIP_OV_H_
