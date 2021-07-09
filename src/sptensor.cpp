#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "sptensor.hpp"


void DestroySparseTensor(SparseTensor *X)
{
    AlignedFree(X->dims);
    AlignedFree(X->vals);
    for(int i = 0; i < X->nmodes; i++) {
      AlignedFree(X->cidx[i]);
    }
    AlignedFree(X->cidx);
    AlignedFree(X);
}

void ExportSparseTensor(
  const char* file_path,
  FileFormat f,
  SparseTensor *X)
{
  assert(f == TEXT_FORMAT || f == BINARY_FORMAT);

  IType nmodes = X->nmodes;
  IType nnz = X->nnz;
  char fn[1024];
  FILE* fp = NULL;

  if (file_path == NULL) {
    if (f == TEXT_FORMAT) {
      sprintf(fn, "%lluD_%llu.tns", nmodes, nnz);
    } else if (f == BINARY_FORMAT) {
      sprintf(fn, "%lluD_%llu.bin", nmodes, nnz);
    }
    fp = fopen(fn, "w");
    assert(fp != NULL);
  } else {
    fp = fopen(file_path, "w");
    assert(fp != NULL);
  }

  if (f == TEXT_FORMAT) {
    IType** cidx = X->cidx;
    FType* vals = X->vals;
    for (IType i = 0; i < nnz; i++) {
      for (IType n = 0; n < nmodes; n++) {
        fprintf(fp, "%llu ", (cidx[n][i] + 1));
      }
      fprintf(fp, "%g\n", vals[i]);
    }
  } else if (f == BINARY_FORMAT) {
    // first write the number of modes
    fwrite(&(X->nmodes), sizeof(IType), 1, fp);
    // then, the dimensions
    fwrite(X->dims, sizeof(IType), X->nmodes, fp);
    fwrite(&(X->nnz), sizeof(IType), 1, fp);

    // write the indices and the values
    for(int i = 0; i < X->nmodes; i++) {
      fwrite(X->cidx[i], sizeof(IType), X->nnz, fp);
    }
    fwrite(X->vals, sizeof(FType), X->nnz, fp);
  }

  fclose(fp);
}


void read_tns_dims(
  FILE* fin,
  IType* num_modes,
  IType* nnz,
  IType** dims
)
{
  // count the number of modes
  IType nmodes = 0;
  ssize_t nread;
  char* line = NULL;
  size_t len = 0;
  while((nread = getline(&line, &len, fin)) != -1) {
    // if line is not empty or a comment (#)
    if(nread > 1 && line[0] != '#') {
      char* ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;
  *num_modes = nmodes;


  // Calculate the tensor dimensions
  assert(nmodes <= MAX_NUM_MODES);
  IType* tmp_dims = (IType*) malloc(sizeof(IType) * nmodes);
  assert(tmp_dims);
  for(IType i = 0; i < nmodes; i++) {
    tmp_dims[i] = 0;
  }
  IType tmp_nnz = 0;

  rewind(fin);
  while((nread = getline(&line, &len, fin)) != -1) {
    // if line is not empty or a comment
    if(nread > 1 && line[0] != '#') {
      char* ptr = line;
      for(IType i = 0; i < nmodes; i++) {
        IType index = strtoull(ptr, &ptr, 10);
        if(index > tmp_dims[i]) {
          tmp_dims[i] = index;
        }
      }
      strtod(ptr, &ptr);
      ++tmp_nnz;
    }
  }
  *dims = tmp_dims;
  *nnz = tmp_nnz;

  rewind(fin);
  free(line);
}


void read_tns_data(
  FILE* fin,
  SparseTensor* tensor,
  IType num_modes,
  IType nnz)
{
  IType tmp_nnz = 0;
  ssize_t nread;
  char* line = NULL;
  size_t len = 0;

  while((nread = getline(&line, &len, fin)) != -1) {
    // if line is not empty or a comment
    if(nread > 1 && line[0] != '#') {
      char* ptr = line;
      for(IType i = 0; i < num_modes; i++) {
        tensor->cidx[i][tmp_nnz] = (IType) strtoull(ptr, &ptr, 10) - 1;
      }
      tensor->vals[tmp_nnz] = (FType) strtod(ptr, &ptr);
      ++tmp_nnz;
    }
  }
  assert(tmp_nnz == nnz);

  free(line);
}



void ImportSparseTensor(
  const char* file_path,
  FileFormat f,
  SparseTensor** X_
)
{
  FILE* fp = fopen(file_path, "r");
  assert(fp != NULL);

  IType nnz = 0;
  IType nmodes = 0;
  IType* dims = NULL;
  // FType* vals = NULL;
  // IType** cidx = NULL;

  if (f == TEXT_FORMAT) {
    // read dims and nnz info from file
    read_tns_dims(fp, &nmodes, &nnz, &dims);

    // allocate memory to the data structures
    SparseTensor* ten = (SparseTensor*) AlignedMalloc(sizeof(SparseTensor));
    assert(ten);

    ten->dims = (IType*) AlignedMalloc(sizeof(IType) * nmodes);
    assert(ten->dims);
    ten->cidx = (IType**) AlignedMalloc(sizeof(IType*) * nmodes);
    assert(ten->cidx);
    for(IType i = 0; i < nmodes; i++) {
        ten->cidx[i] = (IType*) AlignedMalloc(sizeof(IType) * nnz);
        assert(ten->cidx[i]);
    }
    ten->vals = (FType*) AlignedMalloc(sizeof(FType) * nnz);
    assert(ten->vals);

    // populate the data structures
    ten->nmodes = nmodes;
    for(IType i = 0; i < nmodes; i++) {
        ten->dims[i] = dims[i];
    }
    ten->nnz = nnz;
    read_tns_data(fp, ten, nmodes, nnz);

    *X_ = ten;

  } else if (f == BINARY_FORMAT) {
    // first read the number of modes
    nmodes = 0;
    fread(&nmodes, sizeof(IType), 1, fp);
    assert(nmodes <= MAX_NUM_MODES);

    // use this information to read the dimensions of the tensor
    IType* dims = (IType*) AlignedMalloc(sizeof(IType) * nmodes);
    assert(dims);
    fread(dims, sizeof(IType), nmodes, fp);
    // read the nnz
    nnz = 0;
    fread(&nnz, sizeof(IType), 1, fp);
    // use this information to read the index and the values
    IType** cidx = (IType**) AlignedMalloc(sizeof(IType*) * nmodes);
    assert(cidx);
    for(IType i = 0; i < nmodes; i++) {
        cidx[i] = (IType*) AlignedMalloc(sizeof(IType) * nnz);
        assert(cidx[i]);
        fread(cidx[i], sizeof(IType), nnz, fp);
    }
    FType* vals = (FType*) malloc(sizeof(FType) * nnz);
    assert(vals);
    fread(vals, sizeof(FType), nnz, fp);

    // create the sptensor
    SparseTensor* ten = (SparseTensor*) AlignedMalloc(sizeof(SparseTensor));
    assert(ten);
    ten->nmodes = nmodes;
    ten->dims = dims;
    ten->nnz = nnz;
    ten->cidx = cidx;
    ten->vals = vals;

    *X_ = ten;
  }

  if(dims) {
    free(dims);
  }

  fclose(fp);
}

void CreateSparseTensor(
  IType nmodes,
  IType* dims,
  IType nnz,
  IType* cidx,
  FType* vals,
  SparseTensor** X_
)
{
  assert(nmodes > 0);
  assert(nnz > 0);
  for(IType n = 0; n < nmodes; n++) {
    assert(dims[n] > 0);
  }
  for(IType n = 0; n < nmodes; n++) {
    for (IType i = 0; i < nnz; i++) {
      assert(cidx[i * nmodes + n] < dims[n]);
    }
  }

  // create tensor
  SparseTensor* X = (SparseTensor*) AlignedMalloc(sizeof(SparseTensor));
  assert(X);
  X->nmodes = nmodes;
  X->nnz = nnz;
  X->dims = (IType*) AlignedMalloc(sizeof(IType) * nmodes);
  assert(X->dims);
  memcpy(X->dims, dims, sizeof(IType) * nmodes);

  X->cidx = (IType**) AlignedMalloc(sizeof(IType*) * nmodes);
  assert(X->cidx);
  for(IType n = 0; n < nmodes; n++) {
    X->cidx[n] = (IType*) AlignedMalloc(sizeof(IType) * nnz);
    assert(X->cidx[n]);
  }
  for(IType i = 0; i < nnz; i++) {
    for(IType n = 0; n < nmodes; n++) {
        X->cidx[n][i] = cidx[i * nmodes + n];
    }
  }

  X->vals = (FType*) AlignedMalloc(sizeof(FType) * nnz);
  assert(X->vals);
  memcpy(X->vals, vals, sizeof(FType) * nnz);

  *X_ = X;
}

SparseTensor * AllocSparseTensor(const int nnz, const int nmodes) {
  SparseTensor * sp = (SparseTensor *) malloc(sizeof(SparseTensor));
    sp->nnz = nnz;
    sp->nmodes = nmodes;

    sp->vals = (FType*)malloc(nnz * sizeof(FType));
    sp->dims = (IType*)malloc(nmodes * sizeof(IType));
    sp->cidx = (IType**)malloc(nmodes * sizeof(IType*));
    for (int m = 0; m < nmodes; ++m) {
        sp->cidx[m] = (IType*)malloc(nnz * sizeof(IType));
    }
    return sp;
}
