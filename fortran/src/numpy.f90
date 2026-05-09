module numpy
   use iso_fortran_env, only: dp => real64
   use iso_c_binding, only: c_double

   implicit none (type, external)

   integer, parameter :: k = kind(0.0_dp)

contains

   !> Write a rank-2 array to disk in NPY format.
   subroutine write_to_npy_file(path, array)
      character(len=*), intent(in) :: path
      real(kind=k), intent(in) :: array(:, :)
      character(len=8) :: dtype
      integer :: io

      ! Open the file for writing (create/overwrite it)
      open(newunit=io, file=path, &
         status="replace", action="write", &
         form="unformatted", access="stream")

      ! Construct the dtype string
      write (dtype, "(A,I0)") "<f", storage_size(1_dp, dp) / 8

      ! Write magic string, version number and array shape and format
      write(io) npy_header(dtype(1:len_trim(dtype)), .true., shape(array))

      ! Write the array to disk
      write(io) array

      close(io)

   end subroutine write_to_npy_file

   function npy_header(dtype, fortran_order, array_shape)
      character(len=*), intent(in) :: dtype
      logical, intent(in) :: fortran_order
      integer, intent(in) :: array_shape(:)

      character, parameter :: magic_number = char(int(Z'93'))
      character(len=5), parameter :: magic_string = 'NUMPY'
      character(len=8) :: magic_header, fortran_order_string
      character(len=:), allocatable :: dictionary, padding, npy_header

      integer, parameter :: alignment_size = 64
      character, parameter :: nl = new_line(' ')
      integer :: padding_size, dictionary_size

      magic_header = magic_number // magic_string // achar(3) // achar(0)

      if (fortran_order) then
         fortran_order_string = 'True'
      else
         fortran_order_string = 'False'
      end if

      dictionary = "{'descr':'" // dtype // &
         "','fortran_order':" // fortran_order_string(1:len_trim(fortran_order_string)) // "," // &
         "'shape':" // tuple_to_string(array_shape) // "}"

      padding_size = alignment_size - mod(len(magic_header) + 4 + len(dictionary) + 1, alignment_size)

      padding = repeat(' ', padding_size)

      dictionary_size = len(dictionary) + padding_size + 1

      npy_header = magic_header // integer_to_bytes(dictionary_size) // dictionary // padding // nl

   end function npy_header

   pure function tuple_to_string(tuple) result(string)
      integer, intent(in) :: tuple(:)
      character(len=:), allocatable :: string
      character(len=16) :: buffer

      integer :: i

      string = "("
      buffer = ''

      do i = 1, size(tuple)
         write (buffer, "(I0)") tuple(i)
         string = string // buffer(1:len_trim(buffer)) // ','
         buffer = ''
      end do

      string = string // ")"

   end function tuple_to_string

   pure function integer_to_bytes(value) result(string)
      integer, intent(in) :: value
      character(len=4) :: string

      string = achar(mod(value, 256**1)) // &
         achar(mod(value, 256**2) / 256**1) // &
         achar(mod(value, 256**3) / 256**2) // &
         achar(value / 256**3)

   end function integer_to_bytes

end module numpy
