#include <xerus.h>
#include <chrono>

#include "../../classes/QMC/contractpsihek.cpp"
#include "../../classes/QMC/trialfunctions.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

using namespace xerus;
using xerus::misc::operator<<;



int main(){
	std::string path_T = "../data/T_N2_60_single_small.tensor";
	std::string path_V = "../data/V_N2_60_single_small.tensor";
	size_t shift = 135.0,d = 120, p = 14;
	Tensor T,V;

	size_t test_number = 0;
	size_t test_number2 = 3;
	size_t test_number3 = 0;
	std::vector<size_t> sample = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};

	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/initial_value_rank_5.tttensor",phi);
	//read_from_disc("../data/residual_app_" + std::to_string(2*nob)  +"_benchmark_diag_one_step.tttensor",phi);
	phi /= phi.frob_norm(); //normalize
	project(phi,p,d,1e-4);





	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift);



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






