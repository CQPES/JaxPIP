module jaxpip_ort_mod
    use iso_c_binding
    implicit none

    interface

    subroutine init_onnx_model(model_path) bind(c, name="init_onnx_model")
        use iso_c_binding, only: c_char
        implicit none

        character(kind=c_char), dimension(*), intent(in) :: model_path
    end subroutine init_onnx_model

    subroutine eval_onnx_model(coords, num_atoms, energy, forces) bind(c, name="eval_onnx_model")
        use iso_c_binding, only: c_double, c_int
        implicit none

        real(kind=c_double), intent(in) :: coords(*)
        integer(kind=c_int), intent(in) :: num_atoms
        real(kind=c_double), intent(out) :: energy
        real(kind=c_double), intent(out) :: forces(*)
    end subroutine eval_onnx_model

    subroutine finalize_onnx_model() bind(c, name="finalize_onnx_model")
        implicit none

    end subroutine finalize_onnx_model

    end interface

end module jaxpip_ort_mod
