#ifndef GRAM_HPP_
#define GRAM_HPP_

#include "matrix.hpp"
#include "kruskal_model.hpp"

void destroy_grams(Matrix ** grams, KruskalModel* M);
void init_grams(Matrix *** grams, KruskalModel* M);
void update_gram(Matrix * gram, KruskalModel* M, int mode);

#endif