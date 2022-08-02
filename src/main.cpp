#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include <memory>

#include "poisson_generator.hpp"
#include "common.hpp"
#include "alto.hpp"
#include "cpd.hpp"

#include <unistd.h>
#include <sys/resource.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <sched.h>
#include <numaif.h>

#if ALTO_MASK_LENGTH == 64
    typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
    typedef unsigned __int128 LIType;
#else
    #pragma message("!WARNING! ALTO_MASK_LENGTH invalid. Using default 64-bit.")
    typedef unsigned long long LIType;
#endif

#define error(msg...) do {						\
	char ____buf[128];						\
	snprintf(____buf, 128, msg);					\
	fprintf(stderr, "[%s:%d]: %s\n", __FILE__, __LINE__, ____buf);	\
	exit(-1);							\
} while (0)

void BenchmarkAlto(SparseTensor* X, int max_iters, IType rank,
                   IType seed, int target_mode, int num_partitions);

void RunAltoCheck(SparseTensor* X, IType rank, IType seed,
                  int target_mode, int num_partitions);

void VerifyResult(const char* name, FType* truth, FType* factor, IType size);

static void MakeSparseTensor(int nmodes, IType* dims, double sparsity,
                             IType rank, SparseTensor** X);

static std::vector<IType> ParseDimensions(char* argv, int* nmodes_);
static void PrintVersion(char* call);
static void Usage(char* call);

static void PrintTensorInfo(IType rank, int max_iters, SparseTensor* X);

#ifdef memtrace
static long PrintNodeMem(int node, const char* tag);
#endif

const struct option long_opt[] = {
    {"help",           0, NULL, 'h'},
    {"version",        0, NULL, 'v'},
    {"input",          1, NULL, 'i'},
    {"output",         1, NULL, 'o'},
    {"bin",            1, NULL, 'b'},
    {"rank",           1, NULL, 'r'},
    {"max-iter",       1, NULL, 'm'},
    {"seed",           1, NULL, 'x'},
    {"dims",           1, NULL, 'd'},
    {"target-mode",    1, NULL, 't'},
    {"sparsity",       1, NULL, 's'},
    {"epsilon",        1, NULL, 'e'},
    {"file",           1, NULL, 'f'},
    {"check",          0, NULL, 'c'},
    {"bench",          0, NULL, 'p'},
    {NULL,             0, NULL,    0}
};

const char* const short_opt = "hvi:o:b:r:m:x:d:t:s:e:f:cp";
const char* version_info = "0.1.1";

int main(int argc, char** argv)
{
	// Set up timer
	InitTSC();
	uint64_t ticks_start = 0;
	uint64_t ticks_end = 0;
	double t_read = 0.0;
	double t_write = 0.0;
	double t_create = 0.0;
	double t_cpd = 0.0;

	int max_iters = 10;
	IType rank = 16;
	std::vector<IType> dims;
	int nmodes = 0;
	int target_mode = -1;
	std::string text_file;
	std::string text_file_out;
	std::string binary_file;
	double sparsity = 0.1;
	double epsilon = 1e-5;
	int seed = time(NULL);
	int save_to_file = 0;
    bool do_check = false;
    bool do_mttkrp_bench = false;

	int c = 0;
	while ((c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1) {
		switch (c) {
		case 'h':
			Usage(argv[0]);
			return 0;
		case 'v':
			PrintVersion(argv[0]);
			return 0;
		case 'i':
			text_file = std::string(optarg);
			break;
		case 'o':
			text_file_out = std::string(optarg);
			break;
		case 'b':
			binary_file = std::string(optarg);
			break;
		case 'r':
			rank = (IType)atoll(optarg);
			if (rank < 1) {
				fprintf(stderr, "Invalid -rank: %s.\n", optarg);
			}
			break;
		case 'm':
			max_iters = atoi(optarg);
			if (max_iters < 0) {
				fprintf(stderr, "Invalid -max-iter: %s.\n", optarg);
			}
			break;
		case 'x':
			seed = atoi(optarg);
			if (seed < 0) {
				fprintf(stderr, "Invalid -seed: %s.\n", optarg);
			}
			break;
		case 'd':
			dims = ParseDimensions(optarg, &nmodes);
                        if (dims.empty()) {
				fprintf(stderr, "Invalid -dims: %s.\n", optarg);
			}
			break;
		case 't':
			target_mode = atoi(optarg);
			if (target_mode < -1) {
				fprintf(stderr, "Invalid -target-mode: %s.\n", optarg);
                return -1;
			}
			break;
		case 's':
			sparsity = atof(optarg);
			if (sparsity <= 0.0) {
				fprintf(stderr, "Invalid -sparsity: %s.\n", optarg);
			}
			break;
		case 'e':
			epsilon = atof(optarg);
			if (epsilon <= 0.0) {
				fprintf(stderr, "Invalid -epsilon: %s.\n", optarg);
			}
			break;
		case 'f':
			save_to_file = atoi(optarg);
			if (save_to_file < 0) {
				fprintf(stderr, "Invalid -file: %s.\n", optarg);
			}
			break;
        case 'c':
            do_check = true;
            break;
        case 'p':
            do_mttkrp_bench = true;
            break;
		case ':':
			fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			return -1;
		case '?':
			fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			return -1;
		default:
			Usage(argv[0]);
			return -1;
		} // switch (c)
	} // while ((c = getopt_long(...)))

#ifdef memtrace
	printf("Before any initialization\n");
	long long pre_n0, pre_n1;
	pre_n0 = PrintNodeMem(0, "Active:");
	pre_n1 = PrintNodeMem(1, "Active:");
#endif
	SparseTensor* X = NULL;
	if (!binary_file.empty()) {
		BEGIN_TIMER(&ticks_start);
		ImportSparseTensor(binary_file.c_str(), BINARY_FORMAT, &X);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_read);
		PRINT_TIMER("Reading binary file", t_read);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, TEXT_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Writing to text file", t_write);
		}
	}
	else if (!text_file.empty()) {
		BEGIN_TIMER(&ticks_start);
		ImportSparseTensor(text_file.c_str(), TEXT_FORMAT, &X);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_read);
		PRINT_TIMER("Reading text file", t_read);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, BINARY_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Writing to binary file", t_write);
		}
	}
	else if (dims.empty()) {
		fprintf(stderr, "No dims specified... exiting\n");
		Usage(argv[0]);
		exit(-1);
	}
	else {
		BEGIN_TIMER(&ticks_start);
		MakeSparseTensor(nmodes, &dims[0], sparsity, rank, &X);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_create);
		PRINT_TIMER("Creating a new tensor", t_create);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, TEXT_FORMAT, X);
			ExportSparseTensor(NULL, BINARY_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Wrinting to text/binary files", t_write);

		}
	}
    // if check flag is given, only do this
    if (do_check) {
        RunAltoCheck(X, rank, seed, target_mode, omp_get_max_threads());
        return 0;
    }
    if (do_mttkrp_bench) {
#ifdef memtrace
	printf("After initialization\n");
	long long post_n0, post_n1;
	post_n0 = PrintNodeMem(0, "Active:");
	post_n1 = PrintNodeMem(1, "Active:");
	printf("memory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    	// adjust target mode and number of partitions accordingly
	    BenchmarkAlto(X, max_iters, rank, seed, target_mode, omp_get_max_threads());
#ifdef memtrace
	printf("After compute\n");

	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("memory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	    return 0;
    }

	PrintTensorInfo(rank, max_iters, X);

	// Set up the factor matrices
	KruskalModel* M;
	CreateKruskalModel(X->nmodes, X->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);
	// PrintKruskalModel(M);

	/*BEGIN_TIMER(&ticks_start);
	cpd(X, M, max_iters, epsilon);
	END_TIMER(&ticks_end);
	ELAPSED_TIME(ticks_start, ticks_end, &t_cpd);
	PRINT_TIMER("CPD (COO)", t_cpd);

	ExportKruskalModel(M, text_file_out.c_str());*/

	// Convert COO to ALTO
	AltoTensor<LIType>* AT;
	int num_partitions = omp_get_max_threads();
	create_alto(X, &AT, num_partitions);

    BEGIN_TIMER(&ticks_start);
    cpd_alto(AT, M, max_iters, epsilon);
    // cpd(X, M, max_iters, epsilon);
    END_TIMER(&ticks_end);
    ELAPSED_TIME(ticks_start, ticks_end, &t_cpd);
    PRINT_TIMER("CPD (ALTO)", t_cpd);

    // Cleanup
	DestroySparseTensor(X);
	DestroyKruskalModel(M);
	destroy_alto(AT);

	return 0;
}

void BenchmarkAlto(SparseTensor* X, int max_iters, IType rank,
                   IType seed, int target_mode, int num_partitions)
{
	double wtime_s, wtime;

    PrintTensorInfo(rank, max_iters, X);
#ifdef memtrace
	printf("After function call\n");
	long long pre_n0, pre_n1;
	pre_n0 = node_mem(0, "Active:");
	pre_n1 = node_mem(1, "Active:");
#endif
	// Set up the factor matrices
	KruskalModel* M;
	CreateKruskalModel(X->nmodes, X->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);
	// PrintKruskalModel(M);
#ifdef memtrace
	printf("After KruskalModel\n");
	long long post_n0, post_n1;
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	FType** factors = (FType**)AlignedMalloc(X->nmodes * sizeof(FType*));
	assert(factors);
	for (int m = 0; m < X->nmodes; m++) {
        factors[m] = (FType*)AlignedMalloc(X->dims[m] * rank * sizeof(FType));
		assert(factors[m]);
	}
	// Initialize factors
	for (int m = 0; m < X->nmodes; ++m) {
		ParMemcpy(factors[m], M->U[m], X->dims[m] * rank * sizeof(FType));
	}
#ifdef memtrace
	printf("After factors allocation \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// ---------------------------------------------------------------- //
	// Create ALTO tensor from COO
	AltoTensor<LIType>* AT;
    wtime_s = omp_get_wtime();
	create_alto(X, &AT, num_partitions);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO creation time:   %f\n", wtime);
#ifdef memtrace
	printf("After create_alto \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    DestroyKruskalModel(M);
    DestroySparseTensor(X);
#ifdef memtrace
	printf("After deletion of SparseTensor\n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// set up OpenMP locks
    omp_lock_t* writelocks = NULL;
	//IType max_mode_len = 0;
	//for (IType i = 0; i < M->mode; ++i) {
	//	if (max_mode_len < M->dims[i]) {
	//		max_mode_len = M->dims[i];
	//	}
	//}
	//omp_lock_t* writelocks = (omp_lock_t*)AlignedMalloc(sizeof(omp_lock_t) *
	//	max_mode_len);
	//assert(writelocks);
	//for (IType i = 0; i < max_mode_len; ++i) {
	//	omp_init_lock(&(writelocks[i]));
	//}
	FType** ofibs = NULL;
	create_da_mem(target_mode, rank, AT, &ofibs);

#ifdef memtrace
	printf("After create_da_mem \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    // warmup
	if (target_mode == -1) {
		for (int m = 0; m < AT->nmode; ++m) {
			mttkrp_alto_par(m, factors, rank, AT, writelocks, ofibs);
		}
	} else {
		 mttkrp_alto_par(target_mode, factors, rank, AT, writelocks, ofibs);
	}
	// Do ALTO mttkrp
	wtime_s = omp_get_wtime();
	for (int i = 0; i < max_iters; ++i) {
		if (target_mode == -1) {
			for (int m = 0; m < AT->nmode; ++m) {
				mttkrp_alto_par(m, factors, rank, AT, writelocks, ofibs);
			}
		}
		else {
			mttkrp_alto_par(target_mode, factors, rank, AT, writelocks, ofibs);
		}
	}
	wtime = omp_get_wtime() - wtime_s;
	printf("ALTO runtime:   %f\n", wtime);
#ifdef memtrace
	printf("After mttkrp_alto_par \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// ---------------------------------------------------------------- //
	// Cleanup
	destroy_da_mem(AT, ofibs, rank, target_mode);
	destroy_alto(AT);
	// ---------------------------------------------------------------- //
}

void RunAltoCheck(SparseTensor* X, IType rank, IType seed,
                  int target_mode, int num_partitions)
{
    printf("===Check for mode %d===\n", target_mode);
    double wtime_s, wtime;

    PrintTensorInfo(rank, 1, X);
    // ---------------------------------------------------------------- //
    // Set up the factor matrices
    printf("===Create factor matrices===\n");
    KruskalModel *M;
    CreateKruskalModel(X->nmodes, X->dims, rank, &M);
    KruskalModelRandomInit(M, (unsigned int) seed);

    // Create factors for ground truth and ALTO
    FType **truth = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
    assert(truth);
    FType **factors = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
    assert(factors);
    for(int m = 0; m < X->nmodes; m++) {
        truth[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
        assert(truth[m]);
        factors[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
        assert(factors[m]);
    }
    // Initialize factors
    for(int m = 0; m < X->nmodes; ++m) {
        memcpy(factors[m], M->U[m], X->dims[m] * rank * sizeof(FType));
    }
    // ---------------------------------------------------------------- //
    // Do base mttkrp
    printf("===Do base run===\n");
    if (target_mode == -1) {
        for (int m = 0; m < X->nmodes; ++m) {
            printf("mode %d...", m);
            fflush(stdout);
            mttkrp(X, M, (IType) m);
        }
        printf("\n");
    } else {
        mttkrp(X, M, (IType) target_mode);
    }
    printf("   MTTKRP sequential base run done.\n");
    // Copy to ground truth
    for (int m = 0; m < X->nmodes; ++m) {
        memcpy(truth[m], M->U[m], X->dims[m] * rank * sizeof(FType));
    }
    // ---------------------------------------------------------------- //
    // Create ALTO tensor from COO
    printf("===Create ALTO tensors===\n");
    AltoTensor<LIType> *AT;
    create_alto(X, &AT, num_partitions);

    FType **ofibs = NULL;
	create_da_mem(target_mode, rank, AT, &ofibs);

    printf("===Run ALTO MTTKRP===\n");
    wtime_s = omp_get_wtime();
    if (target_mode == -1) {
        for (int m = 0; m < AT->nmode; ++m) {
            printf("mode %d...", m);
            fflush(stdout);
            mttkrp_alto_par(m, factors, rank, AT, NULL, ofibs);
        }
        printf("\n");
    } else {
        mttkrp_alto_par(target_mode, factors, rank, AT, NULL, ofibs);
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("   ALTO runtime: %f s\n", wtime);
    // ---------------------------------------------------------------- //
    // Verify ALTO
    printf("===Verify ALTO MTTKRP=== (target mode: %d)\n", target_mode);
    for (int m = 0; m < AT->nmode; ++m) {
        printf("mode %d: ", m);
        fflush(stdout);
        VerifyResult("mttkrp_alto", truth[m], factors[m], AT->dims[m] * rank);
    }
    // ---------------------------------------------------------------- //
    // Cleanup
	destroy_da_mem(AT, ofibs, rank, target_mode);
    DestroySparseTensor(X);
    destroy_alto(AT);
    // ---------------------------------------------------------------- //
}

void VerifyResult(const char* name, FType* truth, FType* factor, IType size)
{
	IType cnt = 0;
	if (truth != NULL) {
		for (IType i = 0; i < size; ++i) {
			if (fabs(truth[i] - factor[i]) > 1.0e-6) {
				cnt++;
			}
			// printf("%f %f\n", truth[i], factor[i]);
		}
	}
	else {
		fprintf(stderr, "truth is NULL - nothing to compare against\n");
		exit(-1);
	}

	if (cnt == 0) {
		fprintf(stderr, "Results of %s is correct\n", name);
	}
	else {
		fprintf(stderr, "Results of %s is incorrect by %llu\n", name, cnt);
	}
}

static void MakeSparseTensor(int nmodes, IType* dims, double sparsity,
                             IType rank, SparseTensor** X)
{
	IType tmp = 1;
	for (int m = 0; m < nmodes; m++) {
		tmp = tmp * dims[m];
	}
	IType nnz_before = (IType)(sparsity * tmp);

	SparseTensor* X_ = NULL;
	KruskalModel* M_true = NULL;

	// ---------------------------------------------------------------- //
	// Create a Poisson distribution random data
	PoissonGenerator* pg;
	CreatePoissonGenerator(nmodes, dims, &pg);
	IType num_edges = nnz_before;
	PoissonGeneratorRun(pg, num_edges, rank, &M_true, &X_);
	DestroyPoissonGenerator(pg);
	// ---------------------------------------------------------------- //

	DestroyKruskalModel(M_true);
	*X = X_;
}

static std::vector<IType> ParseDimensions(char* argv, int* nmodes_)
{
	std::vector<IType> dims;
	std::string dstr(argv);
	std::string::size_type pos = 0, end = 0;

	while (end != std::string::npos) {
		end = dstr.find(',', pos);

                auto count = end == std::string::npos ? end : end - pos;
		IType d = (IType)stoll(dstr.substr(pos, count));
		if (d >= 0)
			dims.push_back(d);
		pos = end + 1;
	}

	*nmodes_ = dims.size();
	return dims;
}

static void PrintVersion(char* call)
{
	fprintf(stdout, "%s version %s\n", call, version_info);
}

static void Usage(char* call)
{
	fprintf(stderr, "Usage: %s [OPTIONS]\n", call);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h or --help         Display this information\n");
	fprintf(stderr, "\t-v or --version      Display version information\n");
	fprintf(stderr, "\t-i or --input        Input tensor file in text\n");
	fprintf(stderr, "\t-o or --output       Output tensor to file\n");
	fprintf(stderr, "\t-b or --bin          Input tensor file in binary\n");
	fprintf(stderr, "\t-f or --file         Save tensor to another format\n");
	fprintf(stderr, "\t-r or --rank         Rank\n");
	fprintf(stderr, "\t-m or --max-iter     Maximum outter iterations\n");
	fprintf(stderr, "\t-t or --target-mode  Target mode of tensor\n");
	fprintf(stderr, "\t-e or --epsilon      Convergence criteria\n");
	fprintf(stderr, "\t-x or --seed         Random value seed\n");
	fprintf(stderr, "\t-d or --dims         Dimension lenghts (I,J,K)\n");
	fprintf(stderr, "\t-s or --sparsity     Sparsity of generate tensor\n");
    fprintf(stderr, "\t-c or --check        Run ALTO (par) validation against cpd MTTKRP\n");
    fprintf(stderr, "\t-p or --bench        Run ALTO (par) MTTKRP benchmark with the given CMD line options\n");
}

static void PrintTensorInfo(IType rank, int max_iters, SparseTensor* X)
{
	IType* dims = X->dims;
	IType nnz = X->nnz;
	int nmodes = X->nmodes;

	IType tmp = 1;
	for (int i = 0; i < nmodes; i++) {
		tmp *= dims[i];
	}
	double sparsity = ((double)nnz) / tmp;
	fprintf(stderr, "# Modes         = %u\n", nmodes);
	fprintf(stderr, "Rank            = %llu\n", rank);
	fprintf(stderr, "Sparsity        = %f\n", sparsity);
	fprintf(stderr, "Max iters       = %d\n", max_iters);
	fprintf(stderr, "Dimensions      = [%llu", dims[0]);
	for (int i = 1; i < nmodes; i++) {
		fprintf(stderr, " X %llu", dims[i]);
	}
	fprintf(stderr, "]\n");
	fprintf(stderr, "NNZ             = %llu\n", nnz);
}

#ifdef memtrace
static long PrintNodeMem(int node, const char* tag)
{
	// Parse form "Node 0 Active:            43088 kB"
	char name[128], line[1024];
	snprintf(name, 128, "/sys/devices/system/node/node%d/meminfo", node);
	FILE* in = fopen(name, "r");
	assert(in);

	long val = -1;

	while (fgets(line, 1024, in)) {
		char* where = strstr(line, tag);
		printf("%s", line);
		if (!where)
			continue;
		char* ptr = where + strlen(tag);
		while (isspace(*ptr))
			++ptr;
		char* eptr;
		val = strtol(ptr, &eptr, 10);
		if (strstr(eptr, "kB"))
			val *= 1024;
		break;
	}

	// Could keep both files open...
	fclose(in);

	return val;
}
#endif
