#include "stream_matrix.hpp"
#include "util.hpp"

StreamMatrix::StreamMatrix(
    IType rank
) : _nrows(0),
    _ncols(rank),
    _row_capacity(128),
    _mat(NULL)
{
    _mat = init_mat(_row_capacity, rank);
    // _nrows = 12;
    // _ncols = 33;

    printf("=== Stream Matrix initialized ===\n");
}

StreamMatrix::~StreamMatrix()
{
    free_mat(_mat);
}

void StreamMatrix::grow(IType new_rows) {
    if (new_rows < _nrows) {
        // Sufficiently large
        return;
    }
    if (new_rows >= _row_capacity) {
        while(new_rows > _row_capacity) {
            // Increase capacity until number of new rows fits
           _row_capacity *= 2;
        }
        Matrix * newmat = init_mat(_row_capacity, _ncols);
        memcpy(newmat->vals, _mat->vals, _nrows * _ncols * sizeof(*newmat->vals));

        free_mat(_mat);
        _mat = newmat;
    }
    _mat->I = new_rows;
}

void StreamMatrix::grow_rand(IType new_rows) {
    if (new_rows < _nrows) return;
    grow(new_rows);
    if (new_rows >= _nrows) {
        fill_rand(&(_mat->vals[_nrows * _ncols]), (new_rows - _nrows) * _ncols);
    }
    _nrows = new_rows;
}

void StreamMatrix::grow_zero(IType new_rows) {
    if (new_rows < _nrows) return;
    grow(new_rows);
    if (new_rows >= _nrows) {
        for (int i = _nrows * _ncols; i < new_rows * _ncols; ++i) {
            _mat->vals[i] = 0.0;
        }
    }

    _nrows = new_rows;
}
