#include <xerus.h>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/contractpsihek.cpp"


int main(){
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	size_t d = 48,p = 8;
	value_t shift = 25.0;
	std::vector<size_t> hf_sample = {1,2,3,22,23,30,31};
	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";

	auto ehf = makeUnitVector(hf_sample,d);

	ContractPsiHek builder(ehf,d,p,path_T,path_V,0.0, shift);

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	auto lambda = contract_TT(Hs,ehf,ehf);
	TTTensor grad;
	grad(i1&0) = Hs(i1/2,j1/2)*ehf(j1&0);
	grad -= lambda * ehf;
	XERUS_LOG(info,"With Hamiltonian");
	XERUS_LOG(info,grad.frob_norm());
	XERUS_LOG(info,grad.ranks());


	builder.reset(hf_sample);
	auto grad_test = builder.getGrad();
	XERUS_LOG(info,"Without Hamiltonian");
	XERUS_LOG(info,grad_test.frob_norm());
	XERUS_LOG(info,grad_test.ranks());
	XERUS_LOG(info,"Error = " << (grad_test-grad).frob_norm());
	return 0;
}
