#include "jaxpip_ort.h"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <vector>

static Ort::Env* ort_env = nullptr;
static Ort::Session* ort_session = nullptr;
static Ort::MemoryInfo* memory_info = nullptr;

static std::string input_name;
static std::string output_energy_name;
static std::string output_forces_name;

extern "C" {

void init_onnx_model(const char* model_path)
{
    if (ort_session != nullptr) {
        return;
    }

    try {
        ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "JaxPIP_ORT");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        ort_session = new Ort::Session(*ort_env, model_path, session_options);
        memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        Ort::AllocatorWithDefaultOptions allocator;
        input_name = ort_session->GetInputNameAllocated(0, allocator).get();
        output_energy_name = ort_session->GetOutputNameAllocated(0, allocator).get();
        output_forces_name = ort_session->GetOutputNameAllocated(1, allocator).get();

        std::cout << "JaxPIP ONNX model loaded successfully (backend: ONNX Runtime)." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ONNX Runtime initialization error: " << e.what() << std::endl;
    }
}

void eval_onnx_model(double* coords, int* num_atoms, double* energy, double* forces)
{
    if (!ort_session) {
        std::cerr << "Error: Model not initialized! Call init_onnx_model first." << std::endl;
        return;
    }

    int64_t n_atoms = *num_atoms;
    size_t tensor_size = n_atoms * 3;

    std::vector<int64_t> input_shape = { n_atoms, 3 };

    Ort::Value input_tensor = Ort::Value::CreateTensor<double>(
        *memory_info, coords, tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_energy_name.c_str(), output_forces_name.c_str() };

    auto output_tensors = ort_session->Run(
        Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 2);

    double* out_energy = output_tensors[0].GetTensorMutableData<double>();
    *energy = out_energy[0];

    double* out_forces = output_tensors[1].GetTensorMutableData<double>();
    for (size_t i = 0; i < tensor_size; ++i)
        forces[i] = out_forces[i];
}

void finalize_onnx_model()
{
    if (ort_session) {
        delete ort_session;
        ort_session = nullptr;
    }
    if (memory_info) {
        delete memory_info;
        memory_info = nullptr;
    }
    if (ort_env) {
        delete ort_env;
        ort_env = nullptr;
    }
}

} // extern "C"
