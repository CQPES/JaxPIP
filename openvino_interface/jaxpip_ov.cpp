#include "jaxpip_ov.h"

#include <openvino/openvino.hpp>

#include <cstring>
#include <iostream>
#include <memory>

static ov::Core* ov_core = nullptr;
static ov::CompiledModel* ov_compiled_model = nullptr;
static ov::InferRequest* ov_infer_request = nullptr;

extern "C" {

void init_ov_model(const char* model_path)
{
    if (ov_compiled_model != nullptr) {
        return;
    }

    try {
        ov_core = new ov::Core();

        std::shared_ptr<ov::Model> model = ov_core->read_model(model_path);

        ov::AnyMap config = {
            ov::inference_num_threads(1),
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
        };

        ov_compiled_model = new ov::CompiledModel(ov_core->compile_model(model, "CPU", config));

        ov_infer_request = new ov::InferRequest(ov_compiled_model->create_infer_request());

        std::cout << "JaxPIP ONNX model loaded successfully (backend: OpenVINO)." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "OpenVINO initialization error: " << e.what() << std::endl;
    }
}

void eval_ov_model(double* coords, int* num_atoms, double* energy, double* forces)
{
    if (!ov_infer_request) {
        std::cerr << "Error: Model not initialized! Call init_ov_model first." << std::endl;
        return;
    }

    size_t n_atoms = static_cast<size_t>(*num_atoms);
    size_t tensor_size = n_atoms * 3;

    ov::Shape input_shape = { n_atoms, 3 };

    ov::Tensor input_tensor(ov::element::f64, input_shape, coords);

    ov_infer_request->set_input_tensor(0, input_tensor);
    ov_infer_request->infer();

    ov::Tensor energy_tensor = ov_infer_request->get_output_tensor(0);
    const double* out_energy = energy_tensor.data<double>();
    *energy = out_energy[0];

    ov::Tensor forces_tensor = ov_infer_request->get_output_tensor(1);
    const double* out_forces = forces_tensor.data<double>();

    std::memcpy(forces, out_forces, tensor_size * sizeof(double));
}

void finalize_ov_model()
{
    if (ov_infer_request) {
        delete ov_infer_request;
        ov_infer_request = nullptr;
    }
    if (ov_compiled_model) {
        delete ov_compiled_model;
        ov_compiled_model = nullptr;
    }
    if (ov_core) {
        delete ov_core;
        ov_core = nullptr;
    }
}

} // extern "C"
