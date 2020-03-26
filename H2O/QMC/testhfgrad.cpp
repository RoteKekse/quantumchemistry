#include <xerus.h>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/contractpsihek.cpp"


int main(){
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	size_t d = 48,p = 8;
	value_t shift = 25.0;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";

	auto ehf = makeUnitVector(hf_sample,d);

	ContractPsiHek builder(ehf,d,p,path_T,path_V,0.0, shift);
	builder.reset(hf_sample);

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	auto lambda = contract_TT(Hs,ehf,ehf);
	TTTensor grad;
	grad(i1&0) = Hs(i1/2,j1/2)*ehf(j1&0);
	grad -= lambda * ehf;

	XERUS_LOG(info,"With Hamiltonian");
	XERUS_LOG(info,grad.frob_norm());
	grad.round(1e-10);
	XERUS_LOG(info,grad.ranks());

	builder.reset(hf_sample);
	auto grad_test = builder.getGrad();
	grad_test -= lambda * ehf;

	XERUS_LOG(info,"Without Hamiltonian");
	XERUS_LOG(info,grad_test.frob_norm());
	grad_test.round(1e-10);
	XERUS_LOG(info,grad_test.ranks());
	XERUS_LOG(info,"Error = " << (grad_test-grad).frob_norm());

	auto step1 = ehf - 0.3 *grad_test;
	auto step2 = ehf - 0.2 *grad_test;
	auto step3 = ehf - 0.1 *grad_test;
	auto step4 = ehf - 0.05 *grad_test;
	step1 /= step1.frob_norm();
	step2 /= step2.frob_norm();
	step3 /= step3.frob_norm();
	step4 /= step4.frob_norm();
	XERUS_LOG(info,"Start ev " << lambda);
	XERUS_LOG(info,"Step1    " <<contract_TT(Hs,step1,step1));
	XERUS_LOG(info,"Step2    " <<contract_TT(Hs,step2,step2));
	XERUS_LOG(info,"Step3    " <<contract_TT(Hs,step3,step3));
	XERUS_LOG(info,"Step4    " <<contract_TT(Hs,step4,step4));

	size_t rank = 20;
	step1.round(rank);
	step2.round(rank);
	step3.round(rank);
	step4.round(rank);

	step1 /= step1.frob_norm();
	step2 /= step2.frob_norm();
	step3 /= step3.frob_norm();
	step4 /= step4.frob_norm();
	XERUS_LOG(info,"Start ev " << lambda);
	XERUS_LOG(info,"Step1    " <<contract_TT(Hs,step1,step1));
	XERUS_LOG(info,"Step2    " <<contract_TT(Hs,step2,step2));
	XERUS_LOG(info,"Step3    " <<contract_TT(Hs,step3,step3));
	XERUS_LOG(info,"Step4    " <<contract_TT(Hs,step4,step4));

	return 0;
}
