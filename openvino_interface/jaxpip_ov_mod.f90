module jaxpip_ov_mod
    use iso_c_binding
    implicit none

    interface

    subroutine init_ov_model(model_path) bind(c, name="init_ov_model")
        use iso_c_binding, only: c_char
        implicit none
        character(kind=c_char), dimension(*), intent(in) :: model_path
    end subroutine init_ov_model

    subroutine eval_ov_model(coords, num_atoms, energy, forces) bind(c, name="eval_ov_model")
        use iso_c_binding, only: c_double, c_int
        implicit none
        real(kind=c_double), intent(in)  :: coords(*)
        integer(kind=c_int), intent(in)  :: num_atoms
        real(kind=c_double), intent(out) :: energy
        real(kind=c_double), intent(out) :: forces(*)
    end subroutine eval_ov_model

    subroutine finalize_ov_model() bind(c, name="finalize_ov_model")
        implicit none
    end subroutine finalize_ov_model

    end interface

end module jaxpip_ov_mod
