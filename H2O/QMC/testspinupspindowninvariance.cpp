#include <xerus.h>
#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/trialfunctions.cpp"



int main(){
	size_t d = 48,p = 8,iterations = 1e5, rank = 10;
	value_t ev, shift = 25.0, ev_app;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};

	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	auto P = particleNumberOperator(d);
	auto Pup = particleNumberOperatorUp(d);
	auto Pdown = particleNumberOperatorDown(d);

	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));






	return 0;
}
