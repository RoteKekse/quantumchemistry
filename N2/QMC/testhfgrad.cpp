#include <xerus.h>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/contractpsihek.cpp"
#include "../../classes/QMC/tangential_parallel.cpp"


int main(){
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	size_t d = 120,p = 14,rank = 20, iterations=1e5;
	value_t shift = 135.0;

	std::vector<size_t> hf_sample = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};
	std::string path_T = "../data/T_N2_60_single_small.tensor";
	std::string path_V= "../data/V_N2_60_single_small.tensor";

	auto ehf = makeUnitVector(hf_sample,d);

	ContractPsiHek builder(ehf,d,p,path_T,path_V,0.0, shift);
	builder.reset(hf_sample);
	Tangential tang(d,p,iterations,path_T,path_V,shift,hf_sample,ehf);

	auto grad = builder.getGrad(rank);

	auto lambda = tang.get_eigenvalue();
	XERUS_LOG(info,"Start ev " << lambda);

	grad -= lambda * ehf;

	XERUS_LOG(info,"Without Hamiltonian");
	XERUS_LOG(info,grad.frob_norm());
	grad.round(1e-10);
	XERUS_LOG(info,grad.ranks());

	auto step1 = ehf - 0.3 *grad;
	auto step2 = ehf - 0.2 *grad;
	auto step3 = ehf - 0.1 *grad;
	auto step4 = ehf - 0.05 *grad;


	step1.round(rank);
	step2.round(rank);
	step3.round(rank);
	step4.round(rank);

	step1 /= step1.frob_norm();
	step2 /= step2.frob_norm();
	step3 /= step3.frob_norm();
	step4 /= step4.frob_norm();
	XERUS_LOG(info,"Start ev " << lambda);
	tang.update(step1);
	XERUS_LOG(info,"Step1    " <<tang.get_eigenvalue());
	tang.update(step2);
	XERUS_LOG(info,"Step2    " <<tang.get_eigenvalue());
	tang.update(step3);
	XERUS_LOG(info,"Step3    " <<tang.get_eigenvalue());
	tang.update(step4);
	XERUS_LOG(info,"Step4    " <<tang.get_eigenvalue());

	return 0;
}
