#include <xerus.h>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/contractpsihek.cpp"
#include "../../classes/QMC/tangential_parallel.cpp"


int main(){
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	size_t d = 120,p = 14,rank = 60, iterations=1e4;
	value_t shift = 135.0;
	bool build = false;

	std::vector<size_t> hf_sample = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};
	std::string path_T = "../data/T_N2_60_single_small.tensor";
	std::string path_V= "../data/V_N2_60_single_small.tensor";

	auto ehf = makeUnitVector(hf_sample,d);
	auto P = particleNumberOperator(d);

	ContractPsiHek builder(ehf,d,p,path_T,path_V,0.0, shift);
	builder.reset(hf_sample);

	TTTensor grad;
	if (build){
		grad = builder.getGrad(rank);
		Tangential tang(d,p,iterations,path_T,path_V,shift,hf_sample,ehf);
		auto lambda = tang.get_eigenvalue();
		XERUS_LOG(info,"Start ev " << lambda);

		grad -= lambda * ehf;

		XERUS_LOG(info,"Without Hamiltonian");
		XERUS_LOG(info,grad.frob_norm());
		grad.round(1e-10);
		grad.round(rank);
		XERUS_LOG(info,grad.ranks());

		write_to_disc("../data/hf_gradient_120.tttensor",grad);
	}
	else
		read_from_disc("../data/hf_gradient_120.tttensor",grad);

	XERUS_LOG(info,grad.frob_norm());
	auto step1 = ehf - 0.1 *grad;
	XERUS_LOG(info,step1.frob_norm());
	step1.round(20);
	step1 /= step1.frob_norm();
	XERUS_LOG(info,"Particle Number " << contract_TT(P,step1,step1));
	Tangential tang2(d,p,iterations,path_T,path_V,shift,hf_sample,step1);
	XERUS_LOG(info,"Step1    " <<tang2.get_eigenvalue() << "\n" << step1.ranks());

//	auto step1 = ehf - 0.1 *grad;
//	auto step2 = ehf - 0.05 *grad;
//	auto step3 = ehf - 0.02 *grad;
//	auto step4 = ehf - 0.01 *grad;
//	auto step5 = ehf - 0.005 *grad;
//
//
//	step1.round(rank);
//	step2.round(rank);
//	step3.round(rank);
//	step4.round(rank);
//	step5.round(rank);
//
//	step1 /= step1.frob_norm();
//	step2 /= step2.frob_norm();
//	step3 /= step3.frob_norm();
//	step4 /= step4.frob_norm();
//	step5 /= step5.frob_norm();
//	tang.update(step1);
//	XERUS_LOG(info,"Step1    " <<tang.get_eigenvalue() << "\n" << step1.ranks());
//	tang.update(step2);
//	XERUS_LOG(info,"Step2    " <<tang.get_eigenvalue() << "\n" << step2.ranks());
//	tang.update(step3);
//	XERUS_LOG(info,"Step3    " <<tang.get_eigenvalue() << "\n" << step3.ranks());
//	tang.update(step4);
//	XERUS_LOG(info,"Step4    " <<tang.get_eigenvalue() << "\n" << step4.ranks());
//	tang.update(step5);
//	XERUS_LOG(info,"Step5    " <<tang.get_eigenvalue() << "\n" << step5.ranks());

	return 0;
}
