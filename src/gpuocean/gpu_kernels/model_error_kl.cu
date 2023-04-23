/*
This software is part of GPU Ocean

Copyright (C) 2023 SINTEF Digital
Copyright (C) 2023 Norwegian Meteorological Institute

These CUDA kernels generate random fields based on Karhunen-Loeve
basis functions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.cu"


/**
  * Kernel that adds a perturbation to the input field eta.
  * The kernel use Karhunen-Loeve type basis functions with rolling to perturb the eta fields.
  */
extern "C" {
__global__ void kl_sample_eta(
        // Size of computational data
        const int nx_, const int ny_,

        // Parameters related to the KL basis functions
        const int basis_x_start_, const int basis_x_end_,
        const int basis_y_start_, const int basis_y_end_,
        const int include_cos_, const int include_sin_,
        const float kl_decay_, const float kl_scaling_,
        const float roll_x_sin_, const float roll_y_sin_,
        const float roll_x_cos_, const float roll_y_cos_,

        // Normal distributed random numbers of size
        // [basis_x_end - basis_x_start + 1, 2*(basis_y_end - basis_y_start + 1)]
        // (one per KL basis function)
        float* random_ptr_, const int random_pitch_,

        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_
    ) 
    {
        // Each thread is responsible for one grid point in the computational grid.
        
        //Index of cell within block
        const int tx = threadIdx.x; 
        const int ty = threadIdx.y;

        //Index of start of block within domain
        const int bx = blockDim.x * blockIdx.x; // Compansating for ghost cells
        const int by = blockDim.y * blockIdx.y; // Compensating for ghost cells

        //Index of cell within domain
        const int ti = bx + tx;
        const int tj = by + ty;

        // relative location on the unit square
        const float x = (ti + 0.5)/nx_;
        const float y = (tj + 0.5)/ny_;

        const float x_sin = x + roll_x_sin_;
        const float y_sin = y + roll_y_sin_;
        const float x_cos = x + roll_x_cos_;
        const float y_cos = y + roll_y_cos_;
        
        // Shared memory for random numbers
        __shared__ float rns[rand_ny][rand_nx];

        // Load random numbers into shmem
        for (int j = ty; j < rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < rand_nx; i += blockDim.x) {
                rns[j][i] = random_row_[i];
            }
        }
        __syncthreads();

        const int num_basis_x = basis_x_end_ - basis_x_start_ + 1;
        const int num_basis_y = basis_y_end_ - basis_y_start_ + 1;

        // Sample from the KL basis functions
        float d_eta = 0.0f;

        if (include_sin_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    d_eta += kl_scaling_ * rns[j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             sinpif(2*m*y_sin) * sinpif(2*n*x_sin);
                }
            }
        }

        if (include_cos_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    d_eta += kl_scaling_ * rns[num_basis_y + j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             cospif(2*m*y_cos) * cospif(2*n*x_cos);
                }
            }
        }

        if (ti < nx_ && tj < ny_ ) {
            float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj + 2));
            eta_row[ti + 2] += d_eta;
        }
    }

} // extern "C"



/**
  * Kernel that adds a perturbation to the input field eta.
  * The kernel use Karhunen-Loeve type basis functions with rolling to perturb the eta fields.
  */
extern "C" {
__global__ void kl_sample_ocean_state(
        // Size of computational data
        const int nx_, const int ny_,

        // Parameters related to the KL basis functions
        const int basis_x_start_, const int basis_x_end_,
        const int basis_y_start_, const int basis_y_end_,
        const int include_cos_, const int include_sin_,
        const float kl_decay_, const float kl_scaling_,
        const float roll_x_sin_, const float roll_y_sin_,
        const float roll_x_cos_, const float roll_y_cos_,

        // Normal distributed random numbers of size
        // [basis_x_end - basis_x_start + 1, 2*(basis_y_end - basis_y_start + 1)]
        // (one per KL basis function)
        float* random_ptr_, const int random_pitch_,

        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_
    ) 
    {
        // Each thread is responsible for one grid point in the computational grid.
        
        //Index of cell within block
        const int tx = threadIdx.x; 
        const int ty = threadIdx.y;

        //Index of start of block within domain
        const int bx = (blockDim.x-2) * blockIdx.x + 1; // Compansating for ghost cells
        const int by = (blockDim.y-2) * blockIdx.y + 1; // Compensating for ghost cells

        //Index of cell within domain
        const int ti = bx + tx;
        const int tj = by + ty;

        // relative location on the unit square
        const float x = (ti + 0.5 - 2)/nx_;
        const float y = (tj + 0.5 - 2)/ny_;

        const float x_sin = x + roll_x_sin_;
        const float y_sin = y + roll_y_sin_;
        const float x_cos = x + roll_x_cos_;
        const float y_cos = y + roll_y_cos_;
        
        // Shared memory for random numbers
        __shared__ float rns[rand_ny][rand_nx];

        // Shared memory for eta perturbation
        __shared__ float d_eta_shmem[block_height][block_width];

        // Load random numbers into shmem
        for (int j = ty; j < rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < rand_nx; i += blockDim.x) {
                rns[j][i] = random_row_[i];
            }
        }
        __syncthreads();

        const int num_basis_x = basis_x_end_ - basis_x_start_ + 1;
        const int num_basis_y = basis_y_end_ - basis_y_start_ + 1;

        // Sample from the KL basis functions
        float d_eta = 0.0f;

        if (include_sin_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    //d_eta += 1.0f; 
                    d_eta += kl_scaling_ * rns[j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             sinpif(2*m*y_sin) * sinpif(2*n*x_sin);

                }
            }
        }

        if (include_cos_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    //d_eta += 1.0f;
                    d_eta += kl_scaling_ * rns[num_basis_y + j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             cospif(2*m*y_cos) * cospif(2*n*x_cos);
                }
            }
        }

        __syncthreads();
        // Write to shared memory
        //d_eta_shmem[ty][tx] = 1.0f; //d_eta;
    

        __syncthreads();

        if ((tx > 0) && (tx < block_width - 1) && (ty > 0) && (ty < block_height - 1)) {
            if ((ti > 1) && (ti < nx_ + 2) && (tj > 1) && (tj < ny_ +2) ) {
                float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
                eta_row[ti] += d_eta;
            }
        }
    }

} // extern "C"