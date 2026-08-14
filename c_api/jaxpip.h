#ifndef JAXPIP_H_
#define JAXPIP_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the inference runtime and load the JaxPIP ONNX model.
 *        This function should be called ONLY ONCE before the MD loop starts.
 * 
 * @param model_path Path to the exported ONNX model file.
 */
void init_jaxpip_model(const char* model_path);

/**
 * @brief Evaluate the potential energy and analytical forces.
 *        This function can be called repeatedly inside the MD loop.
 * 
 * @param coords    Flattened 1D array of atomic coordinates [x1, y1, z1, ...].
 *                  Size must be 3 * num_atoms.
 * @param num_atoms Pointer to the number of atoms in the system.
 * @param energy    Pointer to store the output potential energy.
 * @param forces    Flattened 1D array to store the output analytical forces.
 *                  Size will be 3 * num_atoms.
 */
void eval_jaxpip_model(const double* coords, const int* num_atoms, double* energy, double* forces);

/**
 * @brief Clean up and release backend runtime resources.
 *        This function should be called at the end of the simulation.
 */
void finalize_jaxpip_model();

#ifdef __cplusplus
}
#endif

#endif // JAXPIP_H_
