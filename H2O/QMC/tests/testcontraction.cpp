#include <xerus.h>
#include "contractpsihek.cpp"
#include "../loading_tensors.cpp"


using namespace xerus;
using xerus::misc::operator<<;


int main(){
	Index i1,i2,j1,j2;
	size_t d = 48;
	size_t p = 10;
	std::string path_T = "../data/T_H2O_"+std::to_string(d)+"_bench.tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(d)+"_bench.tensor";
	value_t nuc = -52.4190597253;

	XERUS_LOG(info, "--- Loading ---");
	XERUS_LOG(info, "Loading Trial Wave Function");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	XERUS_LOG(info, "Loading Hamiltonian Operator");
	TTOperator H;
	read_from_disc("../data/hamiltonian_H2O_48_full_benchmark.ttoperator",H);


	ContractPsiHek builder(phi,d,p,path_T,path_V,nuc);

	XERUS_LOG(info, "--- Running tests ---");
  std::vector<size_t> sample = { 0, 1, 2, 7, 8, 22, 23, 31, 32, 48 };
	auto unit = builder.makeUnitVector(sample);

	XERUS_LOG(info, "Sample: " << sample);
	XERUS_LOG(info,"start contract");
  clock_t begin_time = clock();
	value_t res = builder.contract(sample);
	XERUS_LOG(info,"end contract " << (value_t) (clock() - begin_time) / CLOCKS_PER_SEC << " sec");

	XERUS_LOG(info,"start contract");
	builder.reset(unit);
	value_t d_sample = builder.contract(sample);
	XERUS_LOG(info,"end contract");

	XERUS_LOG(info,"start contract");
	value_t d_sample2 = builder.diagionalEntry(sample);
	XERUS_LOG(info,"end contract");

	Tensor testH,testE;
	testH() = unit(i1&0)*H(i1/2,i2/2)*phi(i2&0);
	testE() = unit(i1&0)*phi(i1&0);

	XERUS_LOG(info,"contracted value for probability:                 " << std::setprecision(10) << testE[0]);
	XERUS_LOG(info,"contracted value for value:                       " << std::setprecision(10) << testH[0]);
	XERUS_LOG(info,"contracted value for sample form loaded operator: " << std::setprecision(10) << testH[0]/testE[0]);
	XERUS_LOG(info,"contracted value for sample from calculation:     " << std::setprecision(10) << res);

	testH() = unit(i1&0)*H(i1/2,i2/2)*unit(i2&0);
	XERUS_LOG(info,"Diagonal entry from operator: " << std::setprecision(10) << testH[0]);
	XERUS_LOG(info,"Diagonal entry from routine:  " << std::setprecision(10) << d_sample);
	XERUS_LOG(info,"Diagonal entry from routine:  " << std::setprecision(10) << d_sample2);

	return 0;
}
