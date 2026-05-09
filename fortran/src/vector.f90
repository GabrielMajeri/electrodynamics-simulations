module vector
   use iso_fortran_env, only: dp => real64

   implicit none (type, external)

   type :: vec3_t
      real(kind=dp) x, y, z
   end type

   type :: vec4_t
      real(kind=dp) a, x, y, z
   end type

   interface operator(+)
      procedure :: add_vec4s
   end interface

   interface operator(*)
      procedure :: multiply_vec4_by_scalar
   end interface

contains

   pure function to_vec4(array) result(v)
      real(kind=dp), intent(in) :: array(1:4)
      type(vec4_t) :: v

      v = vec4_t(a=array(1), x=array(2), y=array(3), z=array(4))

   end function to_vec4

   pure function vec4_to_array(v) result(array)
      type(vec4_t), intent(in) :: v
      real(kind=dp) :: array(1:4)

      array = [v%a, v%x, v%y, v%z]

   end function vec4_to_array

   type(vec4_t) pure function add_vec4s(lhs, rhs) result(res)
      type(vec4_t), intent(in) :: lhs, rhs

      res = vec4_t(lhs%a + rhs%a, lhs%x + rhs%x, lhs%y + rhs%y, lhs%z + rhs%z)

   end function add_vec4s

   type(vec4_t) pure function multiply_vec4_by_scalar(scalar, rhs) result(res)
      real(kind=dp), intent(in) :: scalar
      type(vec4_t), intent(in) :: rhs

      res = vec4_t(scalar * rhs%a, scalar * rhs%x, scalar * rhs%y, scalar * rhs%z)

   end function multiply_vec4_by_scalar

end module vector
