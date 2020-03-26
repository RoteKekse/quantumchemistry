#include <xerus.h>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"


int main(){
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	size_t d = 48;
	size_t p = 8;
	std::vector<size_t> hf_sample = {1,2,3,22,23,30,31};
	auto ehf = makeUnitVector(hf_sample,d);


	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	auto lambda = contract_TT(Hs,ehf,ehf);
	TTTensor grad;
	grad(i1&0) = Hs(i1&0,j1&0)*ehf(j1&0);
	grad -= lambda * ehf;

	XERUS_LOG(info,grad.frob_norm());

	return 0;
}
