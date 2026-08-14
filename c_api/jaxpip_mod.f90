module jaxpip_mod
    use iso_c_binding
    implicit none

    interface

    subroutine init_jaxpip_model(model_path) bind(c, name="init_jaxpip_model")
        use iso_c_binding, only: c_char
        implicit none
        character(kind=c_char), dimension(*), intent(in) :: model_path
    end subroutine init_jaxpip_model

    subroutine eval_jaxpip_model(coords, num_atoms, energy, forces) bind(c, name="eval_jaxpip_model")
        use iso_c_binding, only: c_double, c_int
        implicit none
        real(kind=c_double), intent(in)  :: coords(*)
        integer(kind=c_int), intent(in)  :: num_atoms
        real(kind=c_double), intent(out) :: energy
        real(kind=c_double), intent(out) :: forces(*)
    end subroutine eval_jaxpip_model

    subroutine finalize_jaxpip_model() bind(c, name="finalize_jaxpip_model")
        implicit none
    end subroutine finalize_jaxpip_model

    end interface

end module jaxpip_mod
