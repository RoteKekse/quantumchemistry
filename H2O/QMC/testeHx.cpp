#include <xerus.h>
#include <chrono>

#include "../../classes/QMC/contractpsihek.cpp"
#include "../../classes/QMC/trialfunctions.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

using namespace xerus;
using xerus::misc::operator<<;



int main(){
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_T2 = "../data/T_H2O_48_bench.tensor";
	std::string path_V = "../data/V_H2O_48_bench_single.tensor";
	std::string path_V2 = "../data/V_H2O_48_bench.tensor";
	size_t shift = 25.0,d = 48, p = 8;
	Tensor T,V;
	read_from_disc(path_T2,T);
	read_from_disc(path_V2,V);

	size_t test_number = 1;
	size_t test_number2 = 0;
	std::vector<size_t> sample = {0,1,2,3,22,23,30,31};

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	//read_from_disc("../data/residual_app_" + std::to_string(2*nob)  +"_benchmark_diag_one_step.tttensor",phi);
	phi /= phi.frob_norm(); //normalize
	project(phi,p,d,1e-4);





	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift);

	size_t a = 4, b = 3, c = 2, d = 1;
	std::vector<size_t> idx = {a,b,c,d};
	XERUS_LOG(info,	builder.returnVValue(a,b,c,d) << " " << V[idx]);
)



	for (size_t i = 0; i< test_number; ++i){
		builder.reset(sample);
		value_t val1 = builder.contract();
		auto ek = makeUnitVector(sample,d);
		value_t val2 = contract_TT(Hs,phi,ek);
		XERUS_LOG(info, "Sample = \t" << sample << std::setprecision(12) << " \t"<< std::abs(val1 - val2));
		sample = TrialSample(sample,d);
	}
	auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i< test_number2; ++i){
		builder.reset(sample);
		value_t val1 = builder.contract();
		sample = TrialSample(sample,d);
	}
	auto end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000
		<< " sec");


}






