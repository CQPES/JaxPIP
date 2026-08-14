#ifndef JAXPIP_ORT_H_
#define JAXPIP_ORT_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the ONNX Runtime session and load the JaxPIP model.
 *        This function should be called ONLY ONCE before the MD loop starts.
 * 
 * @param model_path Path to the exported ONNX model file.
 */
void init_onnx_model(const char* model_path);

/**
 * @brief Evaluate the potential energy and analytical forces.
 *        This function can be called repeatedly inside the MD loop.
 * 
 * @param coords    Flattened 1D array of atomic coordinates [x1, y1, z1, x2, y2, z2, ...].
 *                  Size must be 3 * num_atoms.
 * @param num_atoms Pointer to the number of atoms in the system.
 * @param energy    Pointer to store the output potential energy.
 * @param forces    Flattened 1D array to store the output analytical forces.
 *                  Size will be 3 * num_atoms.
 */
void eval_onnx_model(double* coords, int* num_atoms, double* energy, double* forces);

/**
 * @brief Clean up and release ONNX Runtime resources.
 *        This function should be called at the end of the program.
 */
void finalize_onnx_model();

#ifdef __cplusplus
}
#endif

#endif // JAXPIP_ORT_H_
